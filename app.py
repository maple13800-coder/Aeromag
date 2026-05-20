import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import io
import time

# Streamlit 페이지 기본 레이아웃 설정
st.set_page_config(
    page_title="AeroMag - 드론 자력 탐사 데이터 프로세서",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 고급스러운 UI 스타일링을 위한 CSS 주입
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def generate_synthetic_data():
    """
    현실적인 가상의 드론 자력 탐사 데이터를 생성합니다.
    지그재그 비행 경로(Grid)와 함께 아래 요소를 모사합니다:
    - 지하 지질 구조에 의한 자력 이상체 (Dipole anomalies)
    - 센서 헤딩 오차 (방향성 바이어스)
    - 턴어라운드(회전 구간)에서의 고주파 노이즈
    - 비행 속도 변화
    """
    np.random.seed(42)
    
    # 그리드 파라미터
    lines = 11
    points_per_line = 60
    spacing_x = 40.0  # 미터 단위 간격
    spacing_y = 10.0
    
    x_coords = []
    y_coords = []
    timestamps = []
    headings = []  # 비행 방향 (도)
    
    start_time = 1716220800  # Unix 타임스탬프
    current_time = start_time
    
    # 지그재그 경로 생성
    for i in range(lines):
        x = i * spacing_x
        # 방향을 번갈아 가며 왕복 비행 모사
        y_range = range(points_per_line) if i % 2 == 0 else range(points_per_line - 1, -1, -1)
        direction = 0 if i % 2 == 0 else 180  # 북향 vs 남향
        
        for j in y_range:
            y = j * spacing_y
            x_coords.append(x + np.random.normal(0, 0.5))  # 바람에 의한 미세 흔들림
            y_coords.append(y + np.random.normal(0, 0.5))
            timestamps.append(current_time)
            headings.append(direction + np.random.normal(0, 2))
            current_time += 1.0  # 1 Hz 샘플링 속도
            
        # 라인 변경 시 턴어라운드(회전) 구간의 고주파 노이즈 모사
        for turn in range(5):
            x_coords.append(x + spacing_x / 2 + np.random.normal(0, 2))
            y_coords.append((points_per_line if i % 2 == 0 else 0) * spacing_y + np.random.normal(0, 2))
            timestamps.append(current_time)
            headings.append((direction + 90) % 360)
            current_time += 1.0

    df = pd.DataFrame({
        'time': timestamps,
        'x': x_coords,
        'y': y_coords,
        'heading': headings
    })
    
    # 실제 지질 구조 자력값 생성 (두 개의 자력 이상체 모사)
    # 이상체 1: 강한 자성 철광체 (깊은 단일극 소스)
    r1 = np.sqrt((df['x'] - 150)**2 + (df['y'] - 300)**2 + 80**2)
    anomaly_1 = 15000000 / (r1**3)
    
    # 이상체 2: 약한 단층 지질대 (상대적으로 얕은 판상 소스)
    r2 = np.sqrt((df['x'] - 300)**2 + (df['y'] - 150)**2 + 40**2)
    anomaly_2 = -4000000 / (r2**3)
    
    true_earth_field = 48500.0  # 기본 지구 자기장 강도 (nT)
    geology = true_earth_field + anomaly_1 + anomaly_2
    
    # 드론 플랫폼 시스템 노이즈 추가:
    # 1. 헤딩 에러 (비행 방향에 따라 변화하는 기체 자력 영향)
    heading_rad = np.radians(df['heading'])
    heading_error = 15.0 * np.sin(heading_rad) + 5.0 * np.cos(2 * heading_rad)
    
    # 2. 회전(Turnaround) 구간에서의 급격한 가감속 노이즈
    dx = df['x'].diff().fillna(0)
    dy = df['y'].diff().fillna(0)
    dt = df['time'].diff().fillna(1)
    velocity = np.sqrt(dx**2 + dy**2) / dt
    
    acceleration = velocity.diff().fillna(0).abs()
    turn_noise = np.where(acceleration > 2.0, np.random.normal(0, 35, len(df)), np.random.normal(0, 2, len(df)))
    
    # 3. 센서 자체의 가우시안 미세 백색 노이즈
    random_noise = np.random.normal(0, 1.5, len(df))
    
    # 최종 관측 자기장 값 생성
    df['magnetic'] = geology + heading_error + turn_noise + random_noise
    return df

def load_data(file_source):
    """
    사용자가 업로드한 CSV 자력 탐사 데이터를 정형화하여 로드합니다.
    좌표계 명칭과 자력 세기 열을 감지하여 통일시킵니다.
    """
    try:
        df = pd.read_csv(file_source)
        # 다양한 열 이름을 표준 열 이름으로 자동 매핑
        rename_dict = {}
        for col in df.columns:
            col_lower = col.strip().lower()
            if col_lower in ['x', 'east', 'easting', 'longitude', 'lon']:
                rename_dict[col] = 'x'
            elif col_lower in ['y', 'north', 'northing', 'latitude', 'lat']:
                rename_dict[col] = 'y'
            elif col_lower in ['mag', 'magnetic', 'intensity', 'nt', 'field']:
                rename_dict[col] = 'magnetic'
            elif col_lower in ['time', 'timestamp', 'utc']:
                rename_dict[col] = 'time'
            elif col_lower in ['heading', 'dir', 'direction']:
                rename_dict[col] = 'heading'
                
        df = df.rename(columns=rename_dict)
        
        # 필수 칼럼 검증
        required = {'x', 'y', 'magnetic'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"필수 좌표 또는 자력 필드 칼럼이 부족합니다: {missing}")
            
        df['x'] = df['x'].astype(float)
        df['y'] = df['y'].astype(float)
        df['magnetic'] = df['magnetic'].astype(float)
        
        # 시간 정보가 없는 경우 가상 일련번호 부여
        if 'time' not in df.columns:
            df['time'] = np.arange(len(df), dtype=float)
            
        return df, None
    except Exception as e:
        return None, str(e)

def calibrate(df, bias_method="Directional Averaging"):
    """
    드론 본체나 자이로스코프 기울어짐으로 인한 시스템 오차(헤딩 오차)를 보정합니다.
    """
    df_cal = df.copy()
    
    if bias_method == "Mean Subtraction":
        # 전체 데이터의 평균값을 기준값 0으로 리셋 후 오프셋 조정
        mean_field = df_cal['magnetic'].mean()
        df_cal['calibrated_magnetic'] = df_cal['magnetic'] - mean_field
        df_cal['sensor_bias_applied'] = mean_field
        
    elif bias_method == "Directional Averaging":
        # 헤딩 방향이 없는 경우 이동 궤적으로 방향 자동 계산
        if 'heading' not in df_cal.columns:
            dx = df_cal['x'].diff().fillna(0)
            dy = df_cal['y'].diff().fillna(0)
            headings = np.degrees(np.arctan2(dx, dy)) % 360
            df_cal['heading'] = headings
            
        # 4방위(동, 서, 남, 북)로 분할 매핑하는 헬퍼 함수
        def map_heading_to_sector(angle):
            angle = angle % 360
            if (angle >= 315) or (angle < 45):
                return 'North'
            elif (angle >= 45) and (angle < 135):
                return 'East'
            elif (angle >= 135) and (angle < 225):
                return 'South'
            else:
                return 'West'
                
        df_cal['sector'] = df_cal['heading'].apply(map_heading_to_sector)
        
        # 방위별 자기장 평균값과 전체 평균값의 차이를 비교하여 오프셋 산출
        sector_means = df_cal.groupby('sector')['magnetic'].mean()
        global_mean = df_cal['magnetic'].mean()
        
        # 보정 오프셋 계산
        sector_offsets = sector_means - global_mean
        
        # 방위별 보정 적용
        df_cal['sensor_bias_applied'] = df_cal['sector'].map(sector_offsets)
        df_cal['calibrated_magnetic'] = df_cal['magnetic'] - df_cal['sensor_bias_applied']
        
    else:
        df_cal['calibrated_magnetic'] = df_cal['magnetic']
        df_cal['sensor_bias_applied'] = 0.0
        
    return df_cal

def remove_noise(df, thresh_pct=5.0, apply_velocity_filter=True, velocity_thresh=95.0):
    """
    급격한 기체 떨림이나 턴어라운드 시 발생하는 기기 이상 노이즈 데이터를 감지하고 제거합니다.
    - 변화율 기반 필터링 (Difference-based Filter): 인접 데이터 간 자력 변화량이 극단적으로 큰 지점(상위 X%)을 제거합니다.
    - 속도/가속도 필터링 (Velocity Filter): 궤적이 급변하거나 감속이 일어나는 회전 구간의 노이즈를 판단하여 제거합니다.
    """
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # 1. 자력 급변점(스파이크) 노이즈 감지
    mag_diff = df_clean['calibrated_magnetic'].diff().abs().fillna(0)
    cutoff_val = np.percentile(mag_diff, 100.0 - thresh_pct)
    df_clean['mag_spike_filtered'] = mag_diff > cutoff_val
    
    # 2. 회전 제어 및 급감속 필터링
    df_clean['velocity'] = 0.0
    df_clean['accel'] = 0.0
    df_clean['velocity_filtered'] = False
    
    if apply_velocity_filter and len(df_clean) > 2:
        dx = df_clean['x'].diff().fillna(0)
        dy = df_clean['y'].diff().fillna(0)
        dt = df_clean['time'].diff().fillna(1.0)
        dt = np.where(dt <= 0, 1.0, dt)
        
        velocity = np.sqrt(dx**2 + dy**2) / dt
        df_clean['velocity'] = velocity
        
        accel = velocity.diff().abs().fillna(0)
        df_clean['accel'] = accel
        
        accel_cutoff = np.percentile(accel, velocity_thresh)
        df_clean['velocity_filtered'] = accel > accel_cutoff
        
    # 필터 조건 적용
    if apply_velocity_filter:
        valid_indices = (~df_clean['mag_spike_filtered']) & (~df_clean['velocity_filtered'])
    else:
        valid_indices = (~df_clean['mag_spike_filtered'])
        
    df_filtered = df_clean[valid_indices].reset_index(drop=True)
    removed_count = initial_count - len(df_filtered)
    
    return df_filtered, removed_count

def remove_earth_field(df, baseline_method="Median"):
    """
    광역 지구 기본 자기장(베이스라인)을 차감하여, 
    광물 자성 구조를 규명할 수 있는 '잔여 자기 이상(Residual Anomaly)' 값을 추출합니다.
    """
    df_res = df.copy()
    
    if baseline_method == "Median":
        baseline = df_res['calibrated_magnetic'].median()
    elif baseline_method == "Mean":
        baseline = df_res['calibrated_magnetic'].mean()
    else:
        baseline = 0.0
        
    df_res['earth_baseline'] = baseline
    df_res['magnetic_residual'] = df_res['calibrated_magnetic'] - baseline
    
    return df_res

def visualize(df, original_df=None, value_col="magnetic_residual", grid_resolution=100, interpolation_method='cubic'):
    """
    정제된 자력 탐사 데이터를 기반으로 2D 관측 위치 산점도 및 격자화(Grid)된 고품질 등고선 지도를 생성합니다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    
    x = df['x'].values
    y = df['y'].values
    z = df[value_col].values
    
    # 2D 규칙 격자 형성
    xi = np.linspace(x.min(), x.max(), grid_resolution)
    yi = np.linspace(y.min(), y.max(), grid_resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # 불규칙하게 산포된 실측 데이터 경로를 2차원 그리드로 공간 보간 연산 수행
    try:
        zi = griddata((x, y), z, (xi, yi), method=interpolation_method)
    except Exception:
        zi = griddata((x, y), z, (xi, yi), method='nearest')
        interpolation_method = 'nearest'
        
    # 지질학적 자력 대비에 적합한 양방향 대칭 컬러맵(RdBu_r) 지정
    cmap = 'RdBu_r' 
    
    # 좌측: 필터링 및 오차가 정제된 실측 조사 좌표 산점도
    sc = axes[0].scatter(x, y, c=z, cmap=cmap, s=8, alpha=0.9, edgecolor='none')
    axes[0].set_title("정제된 조사 경로 및 실측 포인트 (Cleaned Flight Path)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Easting / X (m)", fontsize=10)
    axes[0].set_ylabel("Northing / Y (m)", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    fig.colorbar(sc, ax=axes[0], label="Magnetic Anomaly (nT)")
    
    # 원본 원시 라인이 있을 경우 옅은 검정 실선으로 뒤배경에 시각화
    if original_df is not None:
        axes[0].plot(original_df['x'], original_df['y'], color='black', alpha=0.15, linestyle='-', linewidth=0.5, label='Original Path', zorder=0)
        axes[0].legend(loc='lower right', fontsize=8)

    # 우측: 공간 보간 알고리즘이 적용된 등고선 분포 지도 (지질학 해석 스탠다드 형태)
    cf = axes[1].contourf(xi, yi, zi, levels=20, cmap=cmap, extend='both')
    contours = axes[1].contour(xi, yi, zi, levels=10, colors='black', linewidths=0.3, alpha=0.7)
    axes[1].clabel(contours, inline=True, fmt='%1.0f', fontsize=8)
    
    axes[1].set_title(f"2차원 지질 공간 등고선 맵 ({interpolation_method.capitalize()} 보간)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Easting / X (m)", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.3)
    fig.colorbar(cf, ax=axes[1], label="Magnetic Residual Anomaly (nT)")
    
    # 축 스케일 동일화 설정
    axes[0].set_aspect('equal', adjustable='box')
    axes[1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig

# 사이드바 - 설정 제어 패널
st.sidebar.image("https://img.icons8.com/color/96/drone.png", width=64)
st.sidebar.markdown("## 제어 및 필터 설정 패널")

st.sidebar.markdown("### 🛠️ 하드웨어 센서 보정")
bias_method = st.sidebar.selectbox(
    "센서 편향(Bias) 완화 모델",
    ["Directional Averaging", "Mean Subtraction", "None"],
    index=0,
    help="Directional Averaging은 실측 비행 방향 벡터를 판단하여 기체 방향 변화에 따른 기생 자기 바이어스를 자동 보정합니다."
)

st.sidebar.markdown("### ⚡ 기체 노이즈 필터링")
spike_thresh = st.sidebar.slider(
    "스파이크 노이즈 허용 필터 (%)",
    min_value=0.0, max_value=20.0, value=5.0, step=0.5,
    help="자력 센서 변화율 중에서 가장 급격하게 도약하는 상위 X%에 해당하는 기체 흔들림 노이즈 데이터를 거릅니다."
)

use_vel_filter = st.sidebar.checkbox("회전 구간(Turnaround) 제거 활성화", value=True)
vel_thresh = st.sidebar.slider(
    "회전 가속도 필터 임계값 (%)",
    min_value=50.0, max_value=99.0, value=95.0, step=1.0,
    disabled=not use_vel_filter,
    help="드론이 왕복 비행을 위해 방향을 90도 이상 꺾을 때 감가속도가 일어나는 영역의 노이즈 데이터를 걸러냅니다."
)

st.sidebar.markdown("### 🌍 지구 광역 배경자기장 제거")
earth_baseline_method = st.sidebar.selectbox(
    "베이스라인 차감 방식",
    ["Median", "Mean", "None"],
    index=0,
    help="주변 넓은 대지의 지자기 바탕 기초값을 차감하여 지하지질체 순수 이상치(Residual)를 부각시킵니다."
)

st.sidebar.markdown("### 🗺️ 격자화 및 렌더링 설정")
grid_res = st.sidebar.slider(
    "보간 격자 밀도 (Grid Density)",
    min_value=50, max_value=250, value=120, step=10,
    help="공간 등고선 지도를 표현하기 위한 가로세로 격자 해상도입니다. 높을수록 매끄럽지만 연산 시간이 늘어납니다."
)
interp_method = st.sidebar.selectbox(
    "공간 격자 보간 알고리즘",
    ["cubic", "linear", "nearest"],
    index=0
)

st.markdown('<div class="main-header">AeroMag 드론 자기 데이터 프로세싱 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">드론 기반 무인 자력 탐사 데이터에서 정밀 물리 신호 보정, 노이즈 필터링 및 2D 매핑 분석을 제공합니다.</div>', unsafe_allow_html=True)

# 작업 탭 구분 구성
tab1, tab2, tab3 = st.tabs(["📊 데이터 수집 및 처리 프로세싱", "🗺️ 2차원 지질 분포도 출력", "📚 지자기 데이터 분석 학술 이론"])

# 세션 상태 초기값 정의
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'synthetic_generated' not in st.session_state:
    st.session_state['synthetic_generated'] = False

with tab1:
    st.markdown("### 📂 자력 탐사 로 데이터 수집")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "자력 탐사 파일(CSV) 업로드", 
            type=["csv"],
            help="구조 좌표 'x', 'y' 및 자기장 세기 필드인 'magnetic' 칼럼이 필수로 요구됩니다."
        )
    
    with col2:
        st.write("✨ **테스트용 탐사 파일이 없으신가요?** 아크로바틱 가상 드론 가상 탐사 에뮬레이터로 예제 데이터를 즉시 만들 수 있습니다:")
        if st.button("가상 탐사 테스트 데이터 에뮬레이트 생성 🚀", use_container_width=True):
            st.session_state['raw_data'] = generate_synthetic_data()
            st.session_state['synthetic_generated'] = True
            st.success("예제 드론 자력 탐사 데이터(11개 왕복 비행선 그리드)가 내부 메모리에 성공적으로 생성되었습니다!")

    if uploaded_file is not None:
        df_loaded, err_msg = load_data(uploaded_file)
        if err_msg:
            st.error(f"CSV 구조 적합성 검증 실패: {err_msg}")
        else:
            st.session_state['raw_data'] = df_loaded
            st.session_state['synthetic_generated'] = False
            st.success("업로드된 사용자 CSV 탐사 데이터를 성공적으로 파싱 및 완료하였습니다.")
            
    if st.session_state['raw_data'] is not None:
        raw_df = st.session_state['raw_data']
        
        st.markdown("---")
        st.markdown("### 📋 수집 자력 데이터 기초 진단 리포트")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 관측 레코드 수", len(raw_df))
        c2.metric("최소 원시값 (nT)", f"{raw_df['magnetic'].min():.1f}")
        c3.metric("최대 원시값 (nT)", f"{raw_df['magnetic'].max():.1f}")
        c4.metric("원시 자기 편차 (std)", f"{raw_df['magnetic'].std():.2f}")
        
        with st.expander("👀 로드된 원본 테이블 데이터 프리뷰 확인 (상위 5개 레코드)"):
            st.dataframe(raw_df.head(), use_container_width=True)
            
        st.markdown("---")
        st.markdown("### ⚙️ 지형-지자기 데이터 보정 연산 실행")
        
        if st.button("물리 센서 보정 및 노이즈 필터링 알고리즘 가동 🚀", type="primary", use_container_width=True):
            with st.spinner("다단 물리 정제 알고리즘 파이프라인 가동 중..."):
                time.sleep(0.5)  # 전문가 스타일의 프로세싱 버퍼 체감용 타임 아웃
                
                # 1단계: 센서 캘리브레이션 및 바이어스 평활화
                cal_df = calibrate(raw_df, bias_method=bias_method)
                
                # 2단계: 신호 전처리 노이즈 거르기
                filtered_df, removed_pts = remove_noise(
                    cal_df, 
                    thresh_pct=spike_thresh, 
                    apply_velocity_filter=use_vel_filter, 
                    velocity_thresh=vel_thresh
                )
                
                # 3단계: 광역 배경 자기 제거를 통한 국소 잔여 이상 수치 계산
                processed_df = remove_earth_field(filtered_df, baseline_method=earth_baseline_method)
                
                st.session_state['processed_data'] = processed_df
                st.session_state['removed_pts_count'] = removed_pts
                
                st.success("지자기 신호 물리 복조 프로세싱이 완전히 완료되었습니다!")
                
        if st.session_state['processed_data'] is not None:
            proc_df = st.session_state['processed_data']
            rem_pts = st.session_state['removed_pts_count']
            
            st.markdown("#### 📈 데이터 정밀 전처리 결과 통계")
            cols = st.columns(4)
            cols[0].metric("정제 후 잔여 탐사 포인트 수", len(proc_df))
            cols[1].metric("기체 흔들림/스파이크 기각 개수", f"{rem_pts}개 ({(rem_pts/len(raw_df))*100:.1f}%)")
            cols[2].metric("잔여 자력이상 최소 (nT)", f"{proc_df['magnetic_residual'].min():.2f}")
            cols[3].metric("잔여 자력이상 최대 (nT)", f"{proc_df['magnetic_residual'].max():.2f}")
            
            with st.expander("👀 보정 처리 완료 테이블 미리보기"):
                st.dataframe(proc_df.head(), use_container_width=True)

with tab2:
    st.markdown("### 🗺️ 지질 구조 자력 분포도 매핑 출력")
    if st.session_state['processed_data'] is not None:
        p_df = st.session_state['processed_data']
        r_df = st.session_state['raw_data']
        
        with st.spinner("지질 자력 공간 매핑 렌더링을 그리는 중..."):
            fig = visualize(
                p_df, 
                original_df=r_df, 
                value_col="magnetic_residual", 
                grid_resolution=grid_res, 
                interpolation_method=interp_method
            )
            st.pyplot(fig)
            
            # 내보내기 기능
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button(
                label="📥 출판용 고해상도 지질 등고선 이미지 다운로드 (PNG)",
                data=buf.getvalue(),
                file_name="geological_magnetic_survey_map.png",
                mime="image/png"
            )
            
            csv_buf = io.StringIO()
            p_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 캘리브레이션 및 노이즈가 제거된 클린 데이터 파일 다운로드 (CSV)",
                data=csv_buf.getvalue(),
                file_name="cleaned_magnetic_data.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ 현재 보정 처리된 자력 탐사 데이터가 부재합니다. 첫 번째 탭에서 데이터 소스를 적재하신 후 '보정 및 필터링 알고리즘 가동' 단추를 눌러 분석을 선행해 주세요.")

with tab3:
    st.markdown(r"""
    ### 📚 항공 자력 탐사 보정 원리 학술 노트
    
    드론을 활용한 자기 센서 비행 측량은 고전적인 도보 측량에 비해 효율적이나, 드론 기체 자체에서 뿜어져 나오는 전자기적 간섭으로 인해 후처리 작업이 반드시 동반됩니다.
    
    #### ⚙️ 1. 센서 캘리브레이션 (헤딩 에러 보정)
    드론 기체의 로터 모터 및 전자 모듈의 전류 흐름은 기생적인 자성 장벽을 만듭니다. 이로 인해 동일한 지점을 비행하더라도 드론이 북쪽을 보고 측정할 때와 남쪽을 보고 측정할 때 센서의 측정값에 편차가 발생하는데, 이를 **헤딩 에러(Heading Error)**라고 지칭합니다.
    - *Directional Averaging*: 궤적 진행 벡터를 판별하여 동/서/남/북 방위각 구간별 센서 고유 오프셋을 계산한 뒤, 전수 보정 오프셋을 차감해 기체 편향 현상을 억제합니다.
    
    #### ⚡ 2. 비행 가속도 역학 노이즈 제거
    드론 측량선 끝자락에서 다음 진행 라인으로 급격히 회전(Turnaround)할 때 롤링/피칭 움직임이 최대화되고 센서에 심각한 관성 왜곡이 동반됩니다.
    - *Spike Reject*: 연속된 신호 간의 점진 도약율($\Delta \text{nT}$) 강도를 평가하여 기계 가속 관성 스파이크를 분리해서 거릅니다.
    - *Velocity Filter*: 순간 이동 속도 변화를 추적하여 물리적 가속 임계 구간인 가상 모서리 측량 지점을 필터 기각 영역으로 판단합니다.
    
    #### 🌍 3. 지구 배경 자기장 제거 (Diurnal & IGRF Background Removal)
    지구 본래의 자기장(국내 기준 통상 48,000 ~ 50,000 nT 수준)은 광물 등 지하지질 이상체의 반응 세기(대개 10 ~ 1,000 nT 내외)에 비해 훨씬 지배적입니다. 따라서 이 배경값을 빼주지 않으면 지하 지질 구조를 시각화할 수 없습니다.
    $$\text{Magnetic Residual (nT)} = \text{Calibrated Reading} - \text{Baseline Field}$$
    
    #### 🗺️ 4. 지질 공간 등고선 보간 (Geological Map Grid Interpolation)
    드론은 탐사 라인을 따라서 좁고 길게 관측 지점을 남기므로, 조사 라인 사이사이의 관측 공백 영역이 매우 넓게 남습니다. 이 이질적인 좌표 포인트들을 **Scipy Griddata** 모듈의 공간 보간 격자화 연산을 이용해 부드러운 다차원 지각 분포로 채워 주어 균일한 2D 지질 해석용 도면을 획득합니다.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("🎯 Developed for Geophysical Flight Operations • v1.2.1")
