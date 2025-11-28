import os
import gc
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import importlib.util
import geopandas as gpd
from matplotlib.path import Path

# 忽略警告
warnings.filterwarnings('ignore')

# ---------------- 配置 ----------------
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'dejavusans'
rcParams['mathtext.default'] = 'rm'

# 路径配置
OBS_DIRS = [r"E:\全国空气质量\站点_20230101-20231231", r"E:\全国空气质量\站点_20240101-20241231"]
SITE_FILE = r"E:\site.xlsx"
MODEL_SFC_DIRS = [r"E:\2023\sfc", r"E:\2024\sfc"]
MODEL_ATMOS_DIRS = [r"E:\2023\atmos", r"E:\2024\atmos"]
CHINA_MAP_DIR = r"C:\Users\lcy\PycharmProjects\PythonProject\chainmap"
CHAP_MODEL_PATH = r"c:\Users\lcy\PycharmProjects\PythonProject\CHAP model.py"
OUT_DIR = "./monthly_mean_contours"

VAR_NAMES = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
START_DATE = pd.Timestamp("2023-10-01")
END_DATE = pd.Timestamp("2024-04-17")

os.makedirs(OUT_DIR, exist_ok=True)

# 全局缓存
OBS_CACHE = {}
MASK_CACHE = {}
STATION_DF_CACHE = None

# ---------------- 地图与掩膜辅助函数 ----------------

def load_china_shapefiles():
    """加载中国地图边界"""
    def try_load(name):
        path = os.path.join(CHINA_MAP_DIR, name)
        if not os.path.exists(path): return None
        for enc in ['gbk', 'utf-8', 'gb18030']:
            try:
                shp = gpd.read_file(path, encoding=enc)
                if shp.crs is None: shp.set_crs(epsg=4326, inplace=True)
                elif shp.crs.to_epsg() != 4326: shp = shp.to_crs(epsg=4326)
                return shp
            except: continue
        return None

    china = try_load('中国行政区划.shp')
    nine = try_load('九段线.shp')
    if china is None: print("⚠️ 无法加载中国地图shapefile")
    return china, None, nine

def create_china_mask_from_shapefile(lon, lat, china_shp):
    """创建中国区域掩膜（已修复MultiPolygon报错）"""
    # 生成缓存键值
    cache_key = (lon.shape, lat.shape, lon[0], lon[-1], lat[0], lat[-1])
    if cache_key in MASK_CACHE:
        return MASK_CACHE[cache_key]

    print("   ⚙️ 正在生成地图掩膜(耗时操作，仅运行一次)...")
    lon_2d, lat_2d = np.meshgrid(lon, lat) if lon.ndim == 1 else (lon, lat)
    
    if china_shp is None:
        mask = (lon_2d >= 73) & (lon_2d <= 135) & (lat_2d >= 18) & (lat_2d <= 54)
    else:
        try:
            geom = china_shp.unary_union.buffer(0.01)
            points = np.column_stack((lon_2d.ravel(), lat_2d.ravel()))
            
            # 【修复】兼容 Polygon 和 MultiPolygon
            if geom.geom_type == 'MultiPolygon':
                polys = list(geom.geoms)
            else:
                polys = [geom]
            
            full_mask_flat = np.zeros(lon_2d.size, dtype=bool)
            
            for poly in polys:
                path = Path(list(poly.exterior.coords))
                is_in = path.contains_points(points)
                full_mask_flat |= is_in
                
                # 处理内部孔洞
                for interior in poly.interiors:
                    path_in = Path(list(interior.coords))
                    is_in_hole = path_in.contains_points(points)
                    full_mask_flat &= ~is_in_hole
            
            mask = full_mask_flat.reshape(lon_2d.shape)

        except Exception as e:
            print(f"   ⚠️ 掩膜生成失败，使用矩形框代替: {e}")
            mask = (lon_2d >= 73) & (lon_2d <= 135) & (lat_2d >= 18) & (lat_2d <= 54)

    MASK_CACHE[cache_key] = mask
    return mask

def _setup_china_map(ax, data_crs, china_shp=None, nineline_shp=None):
    """统一地图设置"""
    # 添加中国边界
    if china_shp is not None:
        china_shp.boundary.plot(ax=ax, transform=data_crs, edgecolor='black', lw=0.8, zorder=10, alpha=0.9)
    else:
        ax.add_feature(cfeature.COASTLINE, lw=0.8)
    
    # 添加九段线
    if nineline_shp is not None:
        nineline_shp.plot(ax=ax, transform=data_crs, edgecolor='#8B0000', lw=1.2, ls='--', zorder=11)

    ax.set_extent([73, 136, 17, 54], crs=data_crs)
    
    # 刻度设置
    xticks = np.arange(75, 136, 10)
    yticks = np.arange(20, 55, 10)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=10)
    
    ax.gridlines(xlocs=xticks, ylocs=yticks, draw_labels=False, 
                 lw=0.5, color='gray', alpha=0.4, ls='--')
    ax.set_aspect(1.2, adjustable='box')

def _below_axes_colorbar_axes(fig, axes, width_scale=0.75, height=0.025, gap=0.04):
    """计算底部色标位置"""
    axes = np.atleast_1d(axes).flatten()
    bboxes = [ax.get_position() for ax in axes]
    left = min(bb.x0 for bb in bboxes)
    right = max(bb.x1 for bb in bboxes)
    bottom = min(bb.y0 for bb in bboxes)
    
    total_width = right - left
    cb_width = total_width * width_scale
    cb_left = left + (total_width - cb_width) / 2
    cb_bottom = bottom - gap - height
    return [cb_left, max(cb_bottom, 0.05), cb_width, height]

# ---------------- 数据处理函数 ----------------

def get_station_info():
    """加载站点信息"""
    global STATION_DF_CACHE
    if STATION_DF_CACHE is None:
        df = pd.read_excel(SITE_FILE, dtype=str)
        df = df.rename(columns={'监测点编码': 'station_id', '经度': 'lon', '纬度': 'lat'})
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        STATION_DF_CACHE = df.dropna(subset=['lon', 'lat'])
    return STATION_DF_CACHE

def read_obs_month_station_means(obs_dirs, month, var):
    """读取观测数据（已修复 KeyError: 'mean'）"""
    key = (month, var)
    if key in OBS_CACHE: return OBS_CACHE[key]

    target_year = month.split('-')[0]
    obs_map = {'PM2.5': ['pm2.5', 'pm25','pm2p5'], 'PM10': ['pm10'], 'CO': ['co'], 
               'NO2': ['no2'], 'SO2': ['so2'], 'O3': ['o3','go3']}
    cands = set(obs_map.get(var.upper(), [var.lower()]))
    
    station_df = get_station_info()
    all_data = []

    for odir in obs_dirs:
        if '202' in odir and target_year not in odir: continue
        if not os.path.exists(odir): continue
        
        month_flat = month.replace('-', '')
        files = [f for f in os.listdir(odir) 
                 if (f.endswith('.csv') or f.endswith('.xlsx')) and month_flat in f]
        
        for fname in files:
            path = os.path.join(odir, fname)
            try:
                if fname.endswith('.csv'):
                    header = pd.read_csv(path, nrows=0, dtype=str)
                    cols = list(header.columns)
                else:
                    header = pd.read_excel(path, nrows=0, dtype=str)
                    cols = list(header.columns)

                if len(cols) < 4: continue

                df = pd.read_csv(path, dtype=str) if fname.endswith('.csv') else pd.read_excel(path, dtype=str)
                df.columns = ['date', 'hour', 'type'] + list(df.columns[3:])
                
                df = df[df['type'].str.lower().isin(cands)]
                if df.empty: continue
                
                df_melt = df.melt(id_vars=['date', 'hour', 'type'], 
                                  value_vars=df.columns[3:],
                                  var_name='station_id', 
                                  value_name='value')
                
                # 原始观测值单位修正：CO 原始数据乘以 1e3，然后再计算月均
                df_melt['value'] = pd.to_numeric(df_melt['value'], errors='coerce')
                if var.upper() == 'CO':
                    df_melt['value'] = df_melt['value'] * 1e3
                all_data.append(df_melt.dropna(subset=['value']))
                
            except Exception:
                continue

    if not all_data:
        return None
    
    full_df = pd.concat(all_data, ignore_index=True)
    res_series = full_df.groupby('station_id')['value'].mean().reset_index()
    
    # 【修复】重命名列
    res_series = res_series.rename(columns={'value': 'mean'})
    
    res = res_series.merge(station_df, on='station_id', how='inner')
    OBS_CACHE[key] = res
    return res

def choose_surface_level(ds, vname, cached_level=None):
    """选择最佳地面层"""
    level_coord = next((c for c in ['pressure_level', 'level', 'lev', 'plev'] if c in ds.coords), None)
    if level_coord is None: return ds, None
    if cached_level is not None: return ds.sel({level_coord: cached_level}), cached_level

    levels = ds[level_coord].values
    lev_hpa = levels / 100.0 if np.max(levels) > 2000 else levels
    
    cand_mask = (lev_hpa >= 850) & (lev_hpa <= 1025)
    if np.any(cand_mask):
        best_lev = levels[cand_mask][np.argmax(lev_hpa[cand_mask])]
    else:
        idx = np.abs(lev_hpa - 925).argmin()
        best_lev = levels[idx]

    print(f"✅ 自动选取层: {best_lev} (Raw Value)")
    return ds.sel({level_coord: best_lev}), best_lev

def compute_model_month_grid_streaming(var, files):
    """流式计算模型月均"""
    lon, lat = None, None
    sums, counts = {}, {}
    cached_lev = None
    target_range_start = START_DATE
    target_range_end = END_DATE
    # 干空气气体常数（用于密度计算）
    R_specific = 287.058

    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                                # 建立一个候选变量名的集合，以更灵活地匹配模型文件中的变量
                candidates = {var.lower(), var.lower().replace('.', '')}
                if var == 'PM2.5':
                    candidates.add('pm2p5')
                if var == 'O3':
                    candidates.add('go3')  # 为 O3 添加 go3 映射

                # 从数据集的变量中寻找第一个匹配的变量名
                vname = next((v for v in ds.data_vars if v.lower() in candidates), None)
                if not vname: continue

                if lon is None:
                    lon = ds[[k for k in ds.coords if 'lon' in k][0]].values
                    lat = ds[[k for k in ds.coords if 'lat' in k][0]].values

                ds_sfc, cached_lev = choose_surface_level(ds, vname, cached_lev)
                time_name = [k for k in ds_sfc.coords if 'time' in k][0]
                
                raw_times = pd.to_datetime(ds_sfc[time_name].values)
                adj_times = raw_times + pd.Timedelta(hours=8)
                
                time_mask = (adj_times >= target_range_start) & (adj_times <= target_range_end)
                if not np.any(time_mask): continue
                
                # 读取当前变量数据（按时间筛选）
                var_data = ds_sfc[vname].isel({time_name: time_mask}).values
                sub_times = adj_times[time_mask]

                # 模型数据单位转换：
                # - PM2.5/PM10：直接乘以 1e9
                # - CO/SO2/NO2/O3：按 ai.py 的公式，将混合比(kg/kg)转换为浓度(μg/m³)
                if var in ['PM2.5', 'PM10']:
                    sub_data = var_data * 1e9
                else:
                    # 寻找温度变量（常见命名：t/temp/temperature）
                    temp_candidates = {'t', 'temp', 'temperature'}
                    temp_vname = next((v for v in ds_sfc.data_vars if v.lower() in temp_candidates), None)
                    if temp_vname is None:
                        # 退一步从原数据集中寻找
                        temp_vname = next((v for v in ds.data_vars if v.lower() in temp_candidates), None)
                    if temp_vname is None:
                        raise RuntimeError('未找到温度变量用于密度计算')

                    T = (ds_sfc[temp_vname] if temp_vname in ds_sfc.data_vars else ds[temp_vname]).isel({time_name: time_mask}).values

                    # 根据选定层的原始数值判断单位：>2000 视为 Pa，否则视为 hPa
                    pressure_pa = cached_lev if (cached_lev is not None and cached_lev > 2000) else (cached_lev * 100 if cached_lev is not None else 100000.0)
                    air_density = pressure_pa / (R_specific * T)
                    sub_data = var_data * air_density * 1e9
                
                months = sub_times.strftime('%Y-%m')
                unique_months = np.unique(months)
                
                for m in unique_months:
                    idx = (months == m)
                    m_data = sub_data[idx]
                    m_sum = np.nansum(m_data, axis=0)
                    m_cnt = np.sum(np.isfinite(m_data), axis=0)
                    
                    if m not in sums:
                        sums[m] = np.zeros_like(m_sum, dtype=np.float32)
                        counts[m] = np.zeros_like(m_cnt, dtype=np.float32)
                    
                    sums[m] += m_sum
                    counts[m] += m_cnt
                    
        except Exception as e:
            print(f"跳过文件 {f}: {e}")
            continue

    grids = {}
    for m in sums:
        with np.errstate(all='ignore'):
            grid = sums[m] / counts[m]
        grid = gaussian_filter(np.nan_to_num(grid), sigma=0.5)
        grids[m] = grid
        
    return grids, lon, lat

def interpolate_obs_to_grid(obs_lon, obs_lat, obs_val, grid_lon, grid_lat):
    """Obs 插值到 Grid (IDW)"""
    if len(obs_val) < 3: return None
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat) if grid_lon.ndim == 1 else (grid_lon, grid_lat)
    
    xy_obs = np.column_stack([obs_lon, obs_lat])
    xy_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    tree = cKDTree(xy_obs)
    dists, idxs = tree.query(xy_grid, k=5)
    weights = 1.0 / (dists**2 + 1e-6)
    vals = np.sum(weights * obs_val[idxs], axis=1) / np.sum(weights, axis=1)
    
    return vals.reshape(grid_x.shape)

def fill_internal_gaps(data, lon, lat):
    """
    【新增】填补内部缺失值 (专门用于修复CHAP数据的方块缺失)
    使用最近邻插值填补 NaN 值
    """
    if data is None: return None
    
    # 找到所有非 NaN 的点
    valid_mask = np.isfinite(data)
    # 如果没有缺失或者全缺失，直接返回
    if np.all(valid_mask) or not np.any(valid_mask):
        return data

    lon_2d, lat_2d = np.meshgrid(lon, lat) if lon.ndim == 1 else (lon, lat)
    
    # 已知点坐标和值
    points = np.column_stack((lon_2d[valid_mask], lat_2d[valid_mask]))
    values = data[valid_mask]
    
    # 待插值点坐标
    target_points = np.column_stack((lon_2d[~valid_mask], lat_2d[~valid_mask]))
    
    if len(points) == 0: return data

    # 使用 nearest 插值填充空洞（适合填补方块状缺失）
    filled_vals = griddata(points, values, target_points, method='nearest')
    
    data_filled = data.copy()
    data_filled[~valid_mask] = filled_vals

    return data_filled

# ---------------- 绘图前处理 ----------------

def _prepare_plot_field(data, lon, lat, china_mask, upscale_factor=2):
    """
    让绘制的填色图紧贴地图边界、边界外留白，并在分辨率较低时进行平滑插值。

    参数
    ----
    data : np.ndarray
        原始待绘制数据（已为中国境外设为 NaN）。
    lon, lat : np.ndarray
        原始经纬度坐标（支持 1D 或 2D）。
    china_mask : np.ndarray
        中国境内为 True 的布尔掩膜。
    upscale_factor : int
        当原始网格较粗时，通过插值将分辨率提升的倍数。
    """
    if data is None:
        return data, lon, lat

    lon_2d, lat_2d = np.meshgrid(lon, lat) if lon.ndim == 1 else (lon, lat)
    valid = np.isfinite(data)

    # 如果数据点过少或无需放大，直接返回掩膜后的数据
    if upscale_factor <= 1 or valid.sum() < 10:
        masked = np.ma.masked_where(~china_mask | ~valid, data)
        return masked, lon, lat

    # 构建更细的经纬度网格
    fine_lon = np.linspace(lon_2d.min(), lon_2d.max(), lon_2d.shape[1] * upscale_factor)
    fine_lat = np.linspace(lat_2d.min(), lat_2d.max(), lat_2d.shape[0] * upscale_factor)
    fine_lon_2d, fine_lat_2d = np.meshgrid(fine_lon, fine_lat)

    # 仅使用有效点进行插值，先用 cubic 提升平滑度，不足部分退回 nearest 以避免空洞
    points = np.column_stack((lon_2d[valid], lat_2d[valid]))
    values = data[valid]
    fine_field = griddata(points, values, (fine_lon_2d, fine_lat_2d), method='cubic')
    fallback = griddata(points, values, (fine_lon_2d, fine_lat_2d), method='nearest')
    fine_field = np.where(np.isfinite(fine_field), fine_field, fallback)

    # 掩膜也插值到高分辨率网格，确保边界紧贴且外部留白
    mask_points = np.column_stack((lon_2d.ravel(), lat_2d.ravel()))
    fine_mask = griddata(mask_points, china_mask.ravel().astype(float),
                         (fine_lon_2d, fine_lat_2d), method='nearest') > 0.5

    masked = np.ma.masked_where(~fine_mask | ~np.isfinite(fine_field), fine_field)
    return masked, fine_lon, fine_lat

# ---------------- 绘图函数 ----------------

def save_plot(fig, path):
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_monthly_mean_triple(aurora, obs, chap, lon, lat, var, save_dir, shapefiles, month_str, china_mask):
    china, nine = shapefiles[0], shapefiles[2]
    data_list = [aurora, obs, chap]
    titles = ['a: Aurora', 'b: Observed', 'c: CHAP']

    # 计算Colorbar范围时忽略NaN
    valid_vals = np.concatenate([d[np.isfinite(d)] for d in data_list])
    if len(valid_vals) == 0: return # 避免空数据报错

    vmin, vmax = np.percentile(valid_vals, [2, 98])
    levels = np.linspace(vmin, vmax, 21)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.15, top=0.90, wspace=0.1)

    cs = None
    for ax, data, title in zip(axes, data_list, titles):
        masked_field, plot_lon, plot_lat = _prepare_plot_field(data, lon, lat, china_mask, upscale_factor=2)
        cs = ax.contourf(plot_lon, plot_lat, masked_field, levels=levels, cmap='viridis', extend='both', transform=ccrs.PlateCarree())
        _setup_china_map(ax, ccrs.PlateCarree(), china, nine)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=5)
        ax.set_facecolor('white')

    cax_rect = _below_axes_colorbar_axes(fig, axes, width_scale=0.6, height=0.03, gap=0.05)
    cax = fig.add_axes(cax_rect)
    cb = fig.colorbar(cs, cax=cax, orientation='horizontal', ticks=levels[::2])
    cb.set_label(f'{var} ($\\mu$g m$^{{-3}}$)', fontsize=12)

    if month_str:
        fig.suptitle(f'{month_str} {var} 月平均浓度', fontsize=18, fontweight='bold', y=0.94)

    save_plot(fig, os.path.join(save_dir, f"{var}_{month_str}_Compare.png"))

def plot_diff_pairs(aurora, obs, chap, lon, lat, var, save_dir, shapefiles, month_str, china_mask):
    china, nine = shapefiles[0], shapefiles[2]
    d1, d2 = aurora - obs, aurora - chap
    
    valid_vals = np.concatenate([d[np.isfinite(d)] for d in [d1, d2]])
    if len(valid_vals) == 0: return

    limit = np.percentile(np.abs(valid_vals), 95)
    levels = np.linspace(-limit, limit, 21)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.15, top=0.90, wspace=0.1)
    
    cs = None
    for ax, data, title in zip(axes, [d1, d2], ['Aurora - Observed', 'Aurora - CHAP']):
        masked_field, plot_lon, plot_lat = _prepare_plot_field(data, lon, lat, china_mask, upscale_factor=2)
        cs = ax.contourf(plot_lon, plot_lat, masked_field, levels=levels, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
        _setup_china_map(ax, ccrs.PlateCarree(), china, nine)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_facecolor('white')
        
    cax_rect = _below_axes_colorbar_axes(fig, axes, width_scale=0.6, height=0.03)
    cax = fig.add_axes(cax_rect)
    cb = fig.colorbar(cs, cax=cax, orientation='horizontal')
    cb.set_label(f'Difference ($\\mu$g m$^{{-3}}$)', fontsize=12)
    
    if month_str:
        # 【修改】标题：差值
        fig.suptitle(f'{month_str} {var} 月平均浓度差值', fontsize=18, fontweight='bold', y=0.94)
        
    save_plot(fig, os.path.join(save_dir, f"{var}_{month_str}_Diff.png"))

def plot_rmse_pairs(aurora, obs, chap, lon, lat, var, save_dir, shapefiles, month_str, china_mask):
    china, nine = shapefiles[0], shapefiles[2]
    # 计算绝对误差
    e1, e2 = np.abs(aurora - obs), np.abs(aurora - chap)
    
    valid_vals = np.concatenate([d[np.isfinite(d)] for d in [e1, e2]])
    if len(valid_vals) == 0: return

    vmax = np.percentile(valid_vals, 98)
    levels = np.linspace(0, vmax, 21)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.15, top=0.90, wspace=0.1)
    
    cs = None
    for ax, data, title in zip(axes, [e1, e2], ['|Aurora - Observed|', '|Aurora - CHAP|']):
        # 色标两端显示倒三角（上下/左右方向的三角形扩展），需将 extend 设置为 'both'
        masked_field, plot_lon, plot_lat = _prepare_plot_field(data, lon, lat, china_mask, upscale_factor=2)
        cs = ax.contourf(plot_lon, plot_lat, masked_field, levels=levels, cmap='inferno_r', extend='both', transform=ccrs.PlateCarree())
        _setup_china_map(ax, ccrs.PlateCarree(), china, nine)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_facecolor('white')
        
    cax_rect = _below_axes_colorbar_axes(fig, axes, width_scale=0.6, height=0.03)
    cax = fig.add_axes(cax_rect)
    cb = fig.colorbar(cs, cax=cax, orientation='horizontal')
    cb.set_label(f'Absolute Error ($\\mu$g m$^{{-3}}$)', fontsize=12)
    
    if month_str:
        # 【修改】标题：RMSE
        fig.suptitle(f'{month_str} {var} 月均RMSE', fontsize=18, fontweight='bold', y=0.94)
        
    save_plot(fig, os.path.join(save_dir, f"{var}_{month_str}_RMSE.png"))

# ---------------- 主程序 ----------------
def main():
    print("📍 初始化...")
    china_shps = load_china_shapefiles()
    
    loader = importlib.machinery.SourceFileLoader('chap_model', CHAP_MODEL_PATH)
    chap_module = loader.load_module()

    months = pd.date_range(START_DATE, END_DATE, freq='MS').strftime('%Y-%m').tolist()

    for var in VAR_NAMES:
        print(f"\n==== 处理变量: {var} ====")
        files = []
        dirs = MODEL_SFC_DIRS if var in ['PM2.5', 'PM10'] else MODEL_ATMOS_DIRS
        for d in dirs:
            if os.path.exists(d):
                files.extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.nc')])
        files = sorted(files)
        
        if not files: continue

        # 1. 计算模型月均
        print("   ▶ 计算模型月均...")
        model_grids, lon, lat = compute_model_month_grid_streaming(var, files)

        # 【修复】如果未能从任何文件中读取到经纬度信息，则跳过当前变量
        if lon is None or lat is None:
            print(f"   ⚠️ 未能从模型文件中加载到 {var} 的有效数据和坐标，跳过该变量。")
            continue
        
        # 生成掩膜 (True代表在中国境内)
        china_mask = create_china_mask_from_shapefile(lon, lat, china_shps[0])
        
        # 2. 逐月处理
        for m in months:
            if m not in model_grids: continue
            
            # 读取观测
            obs_df = read_obs_month_station_means(OBS_DIRS, m, var)
            if obs_df is None: continue
            
            print(f"   ▶ 正在绘图: {m}")
            # 观测 CO 的单位已在原始数据读取阶段完成缩放，此处直接插值月均值
            obs_grid = interpolate_obs_to_grid(obs_df['lon'].values, obs_df['lat'].values, 
                                             obs_df['mean'].values, lon, lat)
            
            # 读取CHAP
            try:
                c_grid, c_lon, c_lat = chap_module.read_chap_month_grid(var, m)
                chap_grid = chap_module.resample_to_model_grid(c_grid, c_lon, c_lat, lon, lat)
            except Exception:
                print(f"     ⚠️ {m} CHAP数据缺失")
                continue

            # ---------------- 数据处理核心修改 ----------------
            
            # CHAP 数据单位修正：仅 CO 需要乘以 1e3
            if var == 'CO' and chap_grid is not None:
                chap_grid = chap_grid * 1e3

            # 1. 填补 CHAP 数据内部空洞 (去方块)
            chap_grid = fill_internal_gaps(chap_grid, lon, lat)
            
            # 2. 对另外两个数据也进行简单填补(可选，防止观测插值有小洞)
            obs_grid = fill_internal_gaps(obs_grid, lon, lat)
            
            # 3. 统一应用中国掩膜：**掩膜外全部置为 NaN** (即白色背景)
            # 注意：np.nan_to_num 会把 NaN 转为 0，这会导致画图变成深蓝色背景，所以这里先不转
            # 我们直接操作原始含 NaN 的数组
            
            # 确保数据是浮点型以便赋值 NaN
            m_grid = model_grids[m].astype(float)
            o_grid = obs_grid.astype(float)
            c_grid = chap_grid.astype(float)

            # 掩膜取反(~)，即中国境外区域 -> 赋值 NaN
            m_grid[~china_mask] = np.nan
            o_grid[~china_mask] = np.nan
            c_grid[~china_mask] = np.nan

            # ------------------------------------------------

            out_path = os.path.join(OUT_DIR, m)
            os.makedirs(out_path, exist_ok=True)

            plot_monthly_mean_triple(m_grid, o_grid, c_grid, lon, lat, var, out_path, china_shps, m, china_mask)
            plot_diff_pairs(m_grid, o_grid, c_grid, lon, lat, var, out_path, china_shps, m, china_mask)
            plot_rmse_pairs(m_grid, o_grid, c_grid, lon, lat, var, out_path, china_shps, m, china_mask)
            
            del obs_df, obs_grid, chap_grid, m_grid, o_grid, c_grid
            gc.collect()

    print("\n✅ 所有任务完成！")

if __name__ == "__main__":
    main()
