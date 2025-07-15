import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import uuid

# Streamlit 앱 제목
st.title("행성과 혜성의 궤도 시뮬레이션")

# 사용자 입력
st.sidebar.header("입력 매개변수")
star_mass = st.sidebar.slider("항성 질량 (태양 질량 단위)", 0.1, 10.0, 1.0)
planet_mass = st.sidebar.slider("행성 질량 (지구 질량 단위)", 0.1, 100.0, 1.0)
comet_mass = st.sidebar.slider("혜성 질량 (지구 질량 단위)", 0.001, 1.0, 0.01)
comet_decay_rate = st.sidebar.slider("혜성 소멸 속도 (질량/초)", 0.0, 0.01, 0.001)
semi_major_axis = st.sidebar.slider("행성 궤도 긴반지름 (AU)", 0.5, 10.0, 1.0)
comet_distance = st.sidebar.slider("혜성-항성 초기 거리 (AU)", 1.0, 20.0, 5.0)

# 상수
G = 6.67430e-11  # 중력 상수 (m^3 kg^-1 s^-2)
M_sun = 1.989e30  # 태양 질량 (kg)
M_earth = 5.972e24  # 지구 질량 (kg)
AU = 1.496e11  # 천문단위 (m)
year = 365.25 * 24 * 3600  # 1년 (초)

# 질량 변환
M_star = star_mass * M_sun
M_planet = planet_mass * M_earth
M_comet = comet_mass * M_earth

# Kepler 제3법칙으로 행성 궤도 주기 계산
def kepler_period(a, M):
    return 2 * np.pi * np.sqrt((a * AU)**3 / (G * M))

# 행성 운동 (원형 궤도 가정)
T_planet = kepler_period(semi_major_axis, M_star)
omega_planet = 2 * np.pi / T_planet

# 혜성 운동 (단순화된 포물선 궤적)
v_comet = np.sqrt(G * M_star / (comet_distance * AU))  # 초기 속도
comet_vx = -v_comet  # 항성 방향으로 초기 속도 설정
comet_vy = 0.0

# 초기 조건
planet_pos = np.array([semi_major_axis * AU, 0.0])
comet_pos = np.array([comet_distance * AU, 0.0])
comet_vel = np.array([comet_vx, comet_vy])

# 운동 방정식 (혜성)
def equations(state, t, M, decay_rate):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax = -G * M * x / r**3
    ay = -G * M * y / r**3
    return [vx, vy, ax, ay]

# 시간 배열
t = np.linspace(0, T_planet, 1000)

# 혜성 궤적 계산
state0 = [comet_pos[0], comet_pos[1], comet_vel[0], comet_vel[1]]
comet_traj = odeint(equations, state0, t, args=(M_star, comet_decay_rate))

# 애니메이션 생성
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-15 * AU, 15 * AU)
ax.set_ylim(-15 * AU, 15 * AU)
ax.set_xlabel("X (AU)")
ax.set_ylabel("Y (AU)")
ax.grid(True)

star, = ax.plot(0, 0, 'yo', markersize=15, label="Star")
planet, = ax.plot([], [], 'bo', markersize=8, label="Planet")
comet, = ax.plot([], [], 'go', markersize=5, label="Comet")
comet_trail, = ax.plot([], [], 'g-', alpha=0.5)
ax.legend()

def init():
    planet.set_data([], [])
    comet.set_data([], [])
    comet_trail.set_data([], [])
    return planet, comet, comet_trail

def animate(i):
    # 행성 위치 (원형 궤도)
    theta = omega_planet * t[i]
    planet_x = semi_major_axis * AU * np.cos(theta)
    planet_y = semi_major_axis * AU * np.sin(theta)
    planet.set_data([planet_x / AU], [planet_y / AU])
    
    # 혜성 위치
    comet_x, comet_y = comet_traj[i, 0], comet_traj[i, 1]
    comet.set_data([comet_x / AU], [comet_y / AU])
    
    # 혜성 궤적
    comet_trail.set_data(comet_traj[:i, 0] / AU, comet_traj[:i, 1] / AU)
    
    # 혜성 질량 감소 효과 (마커 크기 조정)
    current_mass = M_comet * np.exp(-comet_decay_rate * t[i])
    comet.set_markersize(5 * (current_mass / M_comet)**0.5)
    
    return planet, comet, comet_trail

# 애니메이션
ani = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=50, blit=True)

# Streamlit에 애니메이션 표시
st.write("### 궤도 애니메이션")
st.pyplot(fig)

# Matplotlib 애니메이션을 HTML로 변환하여 Streamlit에 표시
from matplotlib.animation import FFMpegWriter
import os
import uuid
video_path = f"animation_{uuid.uuid4()}.mp4"
writer = FFMpegWriter(fps=20)
ani.save(video_path, writer=writer)
st.video(video_path)
os.remove(video_path)  # 임시 파일 삭제
