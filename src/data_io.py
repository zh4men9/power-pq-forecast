"""
Data I/O module for loading and preprocessing power quality data
Automatically detects column names, handles time indexing, resampling, and interpolation
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
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


def load_data(
    file_path: str,
    time_col: Optional[str] = None,
    p_col: Optional[str] = None,
    q_col: Optional[str] = None,
    exog_cols: Optional[list] = None,
    freq: Optional[str] = None,
    tz: Optional[str] = None,
    interp_limit: int = 3
) -> pd.DataFrame:
    """
    Load power quality data from Excel or CSV file
    
    Args:
        file_path: Path to data file
        time_col: Time column name (auto-detect if None)
        p_col: Active power column name (auto-detect if None)
        q_col: Reactive power column name (auto-detect if None)
        exog_cols: List of exogenous variable column names (optional)
        freq: Resampling frequency (e.g., 'H', '15T')
        tz: Timezone
        interp_limit: Maximum consecutive NaN values to interpolate
    
    Returns:
        DataFrame with DatetimeIndex and columns ['P', 'Q', ...exog_cols]
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
    
    # Resample if frequency specified
    if freq:
        df = df.resample(freq).mean()
    
    # Interpolate short gaps only
    if interp_limit > 0:
        for col in df.columns:
            df[col] = df[col].interpolate(method='linear', limit=interp_limit, limit_area='inside')
    
    # Ensure numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values - P: {df['P'].isna().sum()}, Q: {df['Q'].isna().sum()}")
    if exog_cols:
        for col in df.columns:
            if col not in ['P', 'Q']:
                print(f"Missing values - {col}: {df[col].isna().sum()}")
    
    return df


def generate_diagnostic_plots(df: pd.DataFrame, output_dir: str = "outputs/figures"):
    """
    Generate diagnostic plots for data quality inspection
    
    Args:
        df: DataFrame with P and Q columns
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import matplotlib
    from pathlib import Path
    
    # Configure Chinese font - comprehensive list for cross-platform support
    matplotlib.rcParams['font.sans-serif'] = [
        'SimHei',              # Windows
        'Microsoft YaHei',     # Windows
        'STHeiti',             # macOS
        'PingFang SC',         # macOS
        'Heiti SC',            # macOS
        'WenQuanYi Micro Hei', # Linux
        'Noto Sans CJK SC',    # Linux/Cross-platform
        'DejaVu Sans',         # Fallback
        'Arial Unicode MS'     # Fallback
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Overview plot
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
    
    # Missing data heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    
    missing = df.isna().astype(int)
    if missing.sum().sum() > 0:
        im = ax.imshow(missing.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_yticks(range(len(df.columns)))
        ax.set_yticklabels(df.columns, fontsize=10)
        ax.set_xlabel('时间索引', fontsize=12)
        ax.set_title('缺失值分布图', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='缺失(1) / 存在(0)')
    else:
        ax.text(0.5, 0.5, '数据完整，无缺失值', ha='center', va='center', fontsize=16)
        ax.set_title('缺失值分布图', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'missing_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostic plots saved to {output_path}")
