import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from matplotlib import rc

# Matplotlib 애니메이션을 HTML로 렌더링
rc('animation', html='jshtml')

# Streamlit 앱 제목
st.title("행성 또는 혜성 궤도 시뮬레이션")

# 사용자 입력
st.sidebar.header("궤적 선택")
orbit_type = st.sidebar.radio("관찰 대상 선택", ["행성", "혜성"])

# 공통 입력
st.sidebar.header("공통 매개변수")
star_mass = st.sidebar.slider("항성 질량 (태양 질량 단위)", 0.1, 10.0, 1.0, step=0.1)

# 상수
G = 6.67430e-11  # 중력 상수 (m^3 kg^-1 s^-2)
M_sun = 1.989e30  # 태양 질량 (kg)
M_earth = 5.972e24  # 지구 질량 (kg)
AU = 1.496e11  # 천문단위 (m)
year = 365.25 * 24 * 3600  # 1년 (초)

# 질량 변환
M_star = star_mass * M_sun

# 입력 및 계산 로직
if orbit_type == "행성":
    st.sidebar.header("행성 매개변수")
    planet_mass = st.sidebar.slider("행성 질량 (지구 질량 단위)", 0.1, 100.0, 1.0, step=0.1)
    semi_major_axis = st.sidebar.slider("궤도 긴반지름 (AU)", 0.5, 10.0, 1.0, step=0.1)
    
    # Kepler 제3법칙으로 주기 계산
    def kepler_period(a, M):
        return 2 * np.pi * np.sqrt((a * AU)**3 / (G * M))
    
    T = kepler_period(semi_major_axis, M_star)
    omega = 2 * np.pi / T
    M_planet = planet_mass * M_earth
    
else:
    st.sidebar.header("혜성 매개변수")
    comet_mass = st.sidebar.slider("혜성 질량 (지구 질량 단위)", 0.001, 1.0, 0.01, step=0.001)
    comet_distance = st.sidebar.slider("혜성-항성 초기 거리 (AU)", 1.0, 20.0, 5.0, step=0.1)
    comet_decay_rate = st.sidebar.slider("혜성 소멸 속도 (질량/초)", 0.0, 0.01, 0.001, step=0.0001)
    
    # 혜성 초기 조건
    v_comet = np.sqrt(G * M_star / (comet_distance * AU))
    comet_pos = np.array([comet_distance * AU, 0.0])
    comet_vel = np.array([-v_comet, 0.0])
    M_comet = comet_mass * M_earth

# 실행 버튼
if st.button("애니메이션 실행"):
    # 플롯 설정
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-15 * AU, 15 * AU)
    ax.set_ylim(-15 * AU, 15 * AU)
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.grid(True)
    
    star, = ax.plot(0, 0, 'yo', markersize=15, label="Star")
    obj, = ax.plot([], [], 'bo' if orbit_type == "행성" else 'go', markersize=8, label=orbit_type)
    trail, = ax.plot([], [], 'b-' if orbit_type == "행성" else 'g-', alpha=0.5)
    ax.legend()
    
    # 시간 배열
    t = np.linspace(0, T if orbit_type == "행성" else 2 * year, 1000)
    
    # 혜성 운동 방정식
    if orbit_type == "혜성":
        def equations(state, t, M, decay_rate):
            x, y, vx, vy = state
            r = np.sqrt(x**2 + y**2)
            ax = -G * M * x / r**3
            ay = -G * M * y / r**3
            return [vx, vy, ax, ay]
        
        state0 = [comet_pos[0], comet_pos[1], comet_vel[0], comet_vel[1]]
        traj = odeint(equations, state0, t, args=(M_star, comet_decay_rate))
    
    def init():
        obj.set_data([], [])
        trail.set_data([], [])
        return obj, trail
    
    def animate(i):
        if orbit_type == "행성":
            theta = omega * t[i]
            x = semi_major_axis * AU * np.cos(theta)
            y = semi_major_axis * AU * np.sin(theta)
            trail_x = semi_major_axis * AU * np.cos(omega * t[:i])
            trail_y = semi_major_axis * AU * np.sin(omega * t[:i])
            obj.set_data([x / AU], [y / AU])
            trail.set_data(trail_x / AU, trail_y / AU)
        else:
            x, y = traj[i, 0], traj[i, 1]
            current_mass = M_comet * np.exp(-comet_decay_rate * t[i])
            obj.set_data([x / AU], [y / AU])
            obj.set_markersize(8 * (current_mass / M_comet)**0.5)
            trail.set_data(traj[:i, 0] / AU, traj[:i, 1] / AU)
        return obj, trail
    
    # 애니메이션 생성
    ani = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=50, blit=True)
    
    # Streamlit에 애니메이션 표시
    st.write(f"### {orbit_type} 궤적 애니메이션")
    st.pyplot(fig)
    st.write(ani)
