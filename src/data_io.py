"""
Data I/O module for loading and preprocessing power quality data
Automatically detects column names, handles time indexing, resampling, and interpolation
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect time column from common variations
    
    Args:
        df: Input DataFrame
    
    Returns:
        Name of time column or None
    """
    time_keywords = ['时间', '时间戳', '日期', 'time', 'timestamp', 'date', 'datetime', '時間']
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        for keyword in time_keywords:
            if keyword in col_lower:
                return col
    
    return None


def detect_pq_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect P (active power) and Q (reactive power) columns
    
    Args:
        df: Input DataFrame
    
    Returns:
        Tuple of (P column name, Q column name)
    """
    p_keywords = ['有功', 'p', 'active', 'power', '有功功率', 'active_power']
    q_keywords = ['无功', 'q', 'reactive', '無功', '无功功率', 'reactive_power']
    
    p_col = None
    q_col = None
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        # Check for P
        if p_col is None:
            for keyword in p_keywords:
                if keyword in col_lower:
                    p_col = col
                    break
        
        # Check for Q
        if q_col is None:
            for keyword in q_keywords:
                if keyword in col_lower:
                    q_col = col
                    break
    
    return p_col, q_col


def impute_missing_by_nearest_p(df: pd.DataFrame, target_p: float = 280.0) -> pd.DataFrame:
    """
    填补缺失值：使用有功功率接近target_p的行的特征值
    
    Args:
        df: DataFrame with P, Q and other columns
        target_p: 目标有功功率值
    
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    
    # 找到所有包含NaN的行
    rows_with_nan = df[df.isna().any(axis=1)]
    
    if len(rows_with_nan) == 0:
        print("✓ No missing values to impute")
        return df
    
    # 找到完整数据中有功功率最接近target_p的行
    df_complete = df.dropna()
    
    if len(df_complete) == 0:
        print("⚠️  No complete rows available for imputation")
        return df
    
    # 计算与目标值的距离
    distances = np.abs(df_complete['P'] - target_p)
    nearest_idx = distances.idxmin()
    donor_row = df_complete.loc[nearest_idx]
    
    print(f"📌 Imputation donor row: P={donor_row['P']:.2f}, timestamp={nearest_idx}")
    print(f"   Distance from target P={target_p}: {abs(donor_row['P'] - target_p):.2f}")
    
    # 对每个包含NaN的行进行填补
    imputed_count = 0
    for idx in rows_with_nan.index:
        original_row = df.loc[idx].copy()
        na_mask = original_row.isna()
        
        if na_mask.any():
            # 用donor_row的值填补NaN
            df.loc[idx, na_mask] = donor_row[na_mask]
            imputed_count += 1
    
    print(f"✓ Imputed {imputed_count} rows with missing values")
    
    return df


def load_data(
    file_path: str,
    time_col: str = None,
    p_col: str = None,
    q_col: str = None,
    exog_cols: List[str] = None,
    freq: str = 'h',
    tz: str = None,
    interp_limit: int = 3,
    imputation_method: str = None,
    target_p_value: float = 280.0
) -> tuple:
    """
    Load and preprocess power data from CSV file
    
    Args:
        file_path: Path to CSV file
        time_col: Name of time column (None for auto-detect)
        p_col: Name of active power column (None for auto-detect)
        q_col: Name of reactive power column (None for auto-detect)
        exog_cols: List of exogenous variable column names
        freq: Frequency of time series ('h' for hourly)
        tz: Timezone (None for naive datetime)
        interp_limit: Maximum consecutive NaN values to interpolate
        imputation_method: Method for imputing missing values ('nearest_p' or None)
        target_p_value: Target P value for nearest_p imputation
        
    Returns:
        Tuple of (processed_df, original_df_before_imputation)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load file
    if path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif path.suffix == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Detect time column
    if time_col is None:
        time_col = detect_time_column(df)
        if time_col is None:
            raise ValueError("Could not detect time column. Please specify time_col")
        print(f"Auto-detected time column: {time_col}")
    
    # Detect P and Q columns
    if p_col is None or q_col is None:
        detected_p, detected_q = detect_pq_columns(df)
        if p_col is None:
            p_col = detected_p
        if q_col is None:
            q_col = detected_q
        
        if p_col is None or q_col is None:
            raise ValueError("Could not detect P and Q columns. Please specify p_col and q_col")
        
        print(f"Auto-detected P column: {p_col}")
        print(f"Auto-detected Q column: {q_col}")
    
    # Extract relevant columns
    cols_to_keep = [time_col, p_col, q_col]
    
    # Add exogenous columns if specified
    if exog_cols:
        for col in exog_cols:
            if col not in df.columns:
                print(f"Warning: Exogenous column '{col}' not found in data, skipping")
            else:
                cols_to_keep.append(col)
        print(f"Using exogenous columns: {[c for c in exog_cols if c in df.columns]}")
    
    df = df[cols_to_keep].copy()
    
    # Convert time column to datetime
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Set time as index
    df.set_index(time_col, inplace=True)
    
    # Apply timezone if specified
    if tz:
        df.index = df.index.tz_localize(tz)
    
    # Rename columns to standard names
    rename_dict = {p_col: 'P', q_col: 'Q'}
    df.rename(columns=rename_dict, inplace=True)
    
    # Sort by time
    df.sort_index(inplace=True)
    
    # Ensure numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for missing timestamps (gaps in time series)
    original_points = len(df)
    if len(df) > 1:
        time_range = (df.index.max() - df.index.min())
        expected_points = int(time_range.total_seconds() / 3600) + 1  # Assuming hourly data
        missing_timestamps = expected_points - original_points
        
        print(f"Loaded data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        if missing_timestamps > 0:
            print(f"⚠️  Time series has {missing_timestamps} missing timestamps (gaps in data)")
            print(f"   Expected {expected_points} hourly points, found {original_points}")
    
    # Resample to complete hourly frequency (fills missing timestamps with NaN)
    print(f"\n🔄 Resampling to complete hourly frequency...")
    # Use lowercase 'h' to avoid deprecation warning
    freq_str = freq.lower() if freq else 'h'
    df = df.resample(freq_str).asfreq()  # Creates NaN for missing timestamps
    
    print(f"After resampling shape: {df.shape}")
    print(f"Missing values - P: {df['P'].isna().sum()}, Q: {df['Q'].isna().sum()}")
    
    if exog_cols:
        for col in df.columns:
            if col not in ['P', 'Q']:
                print(f"Missing values - {col}: {df[col].isna().sum()}")
    
    # Save copy before imputation for diagnostic plots (with NaN for missing timestamps)
    df_before_imputation = df.copy()
    
    # Apply imputation if specified
    if imputation_method == 'nearest_p':
        print(f"\n🔧 Applying nearest_p imputation (target P={target_p_value})...")
        df = impute_missing_by_nearest_p(df, target_p=target_p_value)
        print(f"After imputation - P: {df['P'].isna().sum()}, Q: {df['Q'].isna().sum()}")
    
    return df, df_before_imputation


def generate_diagnostic_plots(df: pd.DataFrame, df_before: pd.DataFrame = None, output_dir: str = "outputs/figures"):
    """
    Generate diagnostic plots for data quality inspection
    
    Args:
        df: DataFrame with P and Q columns (after processing)
        df_before: DataFrame before imputation (optional, for comparison)
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from pathlib import Path
    
    # Configure matplotlib for Chinese display
    # macOS default Chinese fonts: STHeiti, Heiti TC, PingFang SC, Arial Unicode MS
    plt.rcParams['font.sans-serif'] = ['STHeiti', 'Heiti TC', 'PingFang SC', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['font.monospace'] = ['STHeiti', 'Courier New']  # 等宽字体也支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Overview plot - show before and after imputation if both provided
    if df_before is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Before imputation - P
        axes[0, 0].plot(df_before.index, df_before['P'], linewidth=0.5, alpha=0.8, label='有功功率 P')
        axes[0, 0].set_ylabel('有功功率 P', fontsize=12)
        axes[0, 0].set_title('处理前 - 有功功率', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(loc='best')
        
        # Before imputation - Q  
        axes[1, 0].plot(df_before.index, df_before['Q'], linewidth=0.5, alpha=0.8, color='orange', label='无功功率 Q')
        axes[1, 0].set_ylabel('无功功率 Q', fontsize=12)
        axes[1, 0].set_xlabel('时间', fontsize=12)
        axes[1, 0].set_title('处理前 - 无功功率', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(loc='best')
        
        # After imputation - P
        axes[0, 1].plot(df.index, df['P'], linewidth=0.5, alpha=0.8, label='有功功率 P', color='green')
        axes[0, 1].set_ylabel('有功功率 P', fontsize=12)
        axes[0, 1].set_title('处理后 - 有功功率', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(loc='best')
        
        # After imputation - Q
        axes[1, 1].plot(df.index, df['Q'], linewidth=0.5, alpha=0.8, color='red', label='无功功率 Q')
        axes[1, 1].set_ylabel('无功功率 Q', fontsize=12)
        axes[1, 1].set_xlabel('时间', fontsize=12)
        axes[1, 1].set_title('处理后 - 无功功率', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(loc='best')
        
        fig.suptitle('电力数据总览 - 处理前后对比', fontsize=16, fontweight='bold', y=0.995)
    else:
        # Original single view
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(df.index, df['P'], linewidth=0.5, alpha=0.8, label='有功功率 P')
        axes[0].set_ylabel('有功功率 P', fontsize=12)
        axes[0].set_title('电力数据总览', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')
        
        axes[1].plot(df.index, df['Q'], linewidth=0.5, alpha=0.8, color='orange', label='无功功率 Q')
        axes[1].set_ylabel('无功功率 Q', fontsize=12)
        axes[1].set_xlabel('时间', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path / 'data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Check for time gaps
    time_diffs = df.index.to_series().diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
    
    # Missing data visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    
    missing = df.isna().astype(int)
    if missing.sum().sum() > 0:
        im = ax.imshow(missing.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_yticks(range(len(df.columns)))
        ax.set_yticklabels(df.columns, fontsize=10)
        ax.set_xlabel('时间索引', fontsize=12)
        ax.set_title('字段缺失值分布图', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='缺失(1) / 存在(0)')
    elif len(large_gaps) > 0:
        # Show time gaps info
        gap_info = f'数据字段完整，但时间序列有 {len(large_gaps)} 处时间间隙\n'
        gap_info += f'最大间隙: {large_gaps.max()}\n'
        gap_info += '间隙位置:\n'
        for idx, gap in large_gaps.head(5).items():
            gap_info += f'  {idx}: {gap}\n'
        if len(large_gaps) > 5:
            gap_info += f'  ... 还有 {len(large_gaps)-5} 处间隙'
        
        ax.text(0.5, 0.5, gap_info, ha='center', va='center', fontsize=12, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('时间间隙分析', fontsize=14, fontweight='bold')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, '数据完整，无缺失值', ha='center', va='center', fontsize=16)
        ax.set_title('缺失值分布图', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'missing_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostic plots saved to {output_path}")
