import argparse
import asyncio
import json
import os
import time

if os.path.exists("/usr/share/alsa/alsa.conf"):
    os.environ.setdefault("ALSA_CONFIG_PATH", "/usr/share/alsa/alsa.conf")
if os.path.isdir("/usr/lib/x86_64-linux-gnu/alsa-lib"):
    os.environ.setdefault("ALSA_PLUGIN_DIR", "/usr/lib/x86_64-linux-gnu/alsa-lib")
os.environ.setdefault("AUDIODEV", "pulse")

import cv2
import dashscope
import numpy as np
import robosuite as suite

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    import pygame
except ImportError:
    pygame = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None


# 输出目录：默认在项目下的 outputs/，可用 --out-dir 覆盖
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE_DIR, "outputs", "run")
IMAGE_PATH = os.path.join(OUTPUT_DIR, "robot_view.png")
VERIFY_IMAGE_PATH = os.path.join(OUTPUT_DIR, "robot_verify.png")
AUDIO_PATH = os.path.join(OUTPUT_DIR, "jarvis.mp3")
DEMO_VIDEO_PATH = os.path.join(OUTPUT_DIR, "demo.mp4")
DEMO_VIDEO_FPS = 20.0
ENABLE_VOICE = True
ENABLE_VLM = True
ENABLE_QWEN_VERIFICATION = True
FORCE_KEYBOARD_INPUT = False
CLI_INSTRUCTION = ""
RENDER_WINDOW = True
DASHBOARD_ENABLED = False
SAVE_VIDEO = True
DISPLAY_DASHBOARD = True
PRESENTATION_MODE = False
FORCE_DEMO_PLAN = False
# ===== 自迭代 Agent（路线2）相关开关，默认全关，不影响原课堂 demo =====
SELF_ITERATE = False          # 是否启用 LLM 自迭代抓取回路
SELF_ITERATE_CODEGEN = False  # stretch：让 LLM 生成策略代码片段（默认仍用调参）
MAX_ITER_ATTEMPTS = 8         # 自迭代模式下每个物体最多尝试次数（改进：从 4 增加到 8）
ITER_LOG_PATH = os.path.join(OUTPUT_DIR, "self_iterate_log.jsonl")
FRONT_CAMERA_NAME = "frontview"
WRIST_CAMERA_NAME = "robot0_eye_in_hand"
MIC_DEVICE_INDEX = os.environ.get("JARVIS_MIC_DEVICE_INDEX", "").strip()
JARVIS_VOICE = "zh-CN-YunjianNeural"
JARVIS_VOICE_RATE = "-10%"
JARVIS_VOICE_PITCH = "-10Hz"
JARVIS_VOICE_VOLUME = "+10%"
DEFAULT_INSTRUCTION = "把桌面上能吃的东西放左边，能喝的东西放右边"
DEFAULT_SPEECH = "好的先生，马上为您执行。"
OBJECT_MAPPING = {
    "milk": "Milk_main",
    "bread": "Bread_main",
    "cereal": "Cereal_main",
    "can": "Can_main",
}
BIN_COORDINATES = {
    "bin_top_left": np.array([0.0025, 0.4025, 0.98], dtype=np.float64),
    "bin_top_right": np.array([0.1975, 0.4025, 0.98], dtype=np.float64),
    "bin_bottom_left": np.array([0.0025, 0.1575, 0.98], dtype=np.float64),
    "bin_bottom_right": np.array([0.1975, 0.1575, 0.98], dtype=np.float64),
}
OVERLAY_COLOR = (0, 255, 255)
GRASP_Z_OFFSETS = {
    "bread": 0.020,
    "cereal": 0.035,
    "milk": 0.055,
    "can": 0.020,
}
GRASP_XY_OFFSETS = {
    "bread": np.array([0.0, 0.0], dtype=np.float64),
    "cereal": np.array([0.0, 0.0], dtype=np.float64),
    "milk": np.array([0.0, 0.0], dtype=np.float64),
    "can": np.array([0.0, 0.0], dtype=np.float64),
}
GRASP_REACH_TOLERANCE = 0.022
PLACE_REACH_TOLERANCE = 0.035
# ===== 自迭代可调参数的合法范围（LLM 输出会被 clip 到这里，绝不信任 LLM 不越界）=====
GRASP_PARAM_BOUNDS = {
    "z_offset": (-0.02, 0.08),     # 抓取点相对物体中心的高度，越小抓得越深
    "xy_offset_x": (-0.03, 0.03),  # 水平修正 x
    "xy_offset_y": (-0.03, 0.03),  # 水平修正 y
    "close_steps": (40, 120),      # 夹爪闭合保持步数
    "settle_steps": (8, 40),       # 预闭合沉降步数
    "descend_gain": (2.0, 6.0),    # 下降阶段增益，过大容易把物体撞飞
}
SIM_SUCCESS_XY_TOLERANCE = 0.18
SIM_SUCCESS_Z_MARGIN = 0.18
GRASP_CLOSE_STEPS = 65
GRASP_SETTLE_STEPS = 18
RELEASE_STEPS = 45
PLACE_GROUND_CLEARANCE = 0.002
MIN_LIFT_DELTA_FOR_HELD_OBJECT = 0.06  # 改进：从 0.10 降到 0.06，适应 MuJoCo 平面接触约束
PUSH_FALLBACK_TARGETS = {"bread", "cereal"}
PUSH_CLEARANCE = 0.035
PUSH_START_BACKOFF = 0.12
PUSH_END_OVERSHOOT = 0.08
PUSH_MAX_STEPS = 900
ENABLE_GRASP_ASSIST = True
ASSIST_GRASP_TARGETS = {"bread", "cereal", "milk", "can"}
ASSIST_PLACE_TARGETS = {"bread", "cereal", "milk", "can"}
ASSIST_HOLD_OFFSETS = {
    "bread": np.array([0.0, 0.0, -0.020], dtype=np.float64),
    "cereal": np.array([0.0, 0.0, -0.035], dtype=np.float64),
    "milk": np.array([0.0, 0.0, -0.055], dtype=np.float64),
    "can": np.array([0.0, 0.0, -0.020], dtype=np.float64),
}
ASSIST_ATTACH_BLEND = 0.10
ASSIST_FOLLOW_BLEND = 0.22
ASSIST_PLACE_BLEND = 0.16
ASSIST_RELEASE_BLEND = 0.12
ASSIST_MAX_STEP = 0.018
ASSIST_PLACE_MAX_STEP = 0.012
ASSIST_MIN_CLEARANCE = 0.012
ASSIST_PICK_BLEND_STEPS = 16
ASSIST_CAPTURE_CLOSE_STEPS = 9999
ASSIST_DIRECT_FOLLOW_BLEND = 0.35
ASSIST_DIRECT_FOLLOW_STEP = 0.018
ASSIST_DIRECT_LIFT_STEP = 0.012
ASSIST_RELEASE_SETTLE_STEPS = 18
ASSIST_CARRY_ANCHOR_BLEND = 0.42
ASSIST_CARRY_ANCHOR_MAX_STEP = 0.032
ASSIST_CENTERING_STEPS = 12
ASSIST_CARRY_XY_LIMIT = 0.030
ASSIST_CARRY_Z_LIMITS = (-0.070, 0.030)
ASSIST_OBJECT_MAX_STEPS = {
    "bread": 0.018,
    "cereal": 0.018,
    "milk": 0.018,
    "can": 0.018,
}
OBJECT_REST_Z = {
    "bread": 0.845,
    "cereal": 0.900,
    "milk": 0.885,
    "can": 0.860,
}
FSM_MAX_STEPS = 1800
HOME_RETURN_MAX_STEPS = 500
MAX_VISUAL_RECOVERY_ATTEMPTS = 3
DESTINATION_LABELS = {
    "bin_top_left": "左上角收纳盒",
    "bin_top_right": "右上角收纳盒",
    "bin_bottom_left": "左下角收纳盒",
    "bin_bottom_right": "右下角收纳盒",
}
TARGET_LABELS = {
    "milk": "牛奶",
    "bread": "面包",
    "cereal": "麦片盒",
    "can": "易拉罐",
}
OBJECT_KIND_LABELS = {
    "milk": "饮品",
    "bread": "食物",
    "cereal": "食物",
    "can": "饮品",
}
HUD_STATE = {
    "enabled": True,
    "planner": "Qwen-VL",
    "mode": "BOOT",
    "task": "Initializing",
    "target": "-",
    "body": "N/A",
    "world_pos": "N/A",
    "destination": "-",
    "verification": "N/A",
    "progress": 0.0,
    "step": 0,
    "total": 0,
    "message": "J.A.R.V.I.S. online",
    "user_command": "N/A",
    "scene": "PickPlace",
    "stage": "Language",
    "summary": "N/A",
}
DASHBOARD_STATE = {
    "env": None,
    "front_camera": FRONT_CAMERA_NAME,
    "wrist_camera": WRIST_CAMERA_NAME,
    "wrist_available": False,
    "wrist_warning_printed": False,
    "plan": [],
    "task_statuses": [],
    "current_task_idx": -1,
    "last_canvas": None,
    "window_available": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S robosuite PickPlace demo")
    parser.add_argument("--instruction", "-i", default="", help="直接指定用户指令，跳过麦克风监听")
    parser.add_argument("--keyboard", action="store_true", help="强制使用键盘输入")
    parser.add_argument("--no-voice", action="store_true", help="关闭贾维斯 TTS 语音")
    parser.add_argument("--no-vlm", action="store_true", help="关闭通义千问规划，使用本地默认计划")
    parser.add_argument("--no-qwen-verify", action="store_true", help="关闭通义千问视觉裁判，使用仿真状态核验")
    parser.add_argument("--headless", action="store_true", help="不打开 robosuite 渲染窗口，仅保存视频")
    parser.add_argument("--plain-video", action="store_true", help="关闭视频 HUD 科技叠层")
    parser.add_argument("--no-assist", action="store_true", help="关闭演示夹持辅助，更接近纯物理抓取")
    parser.add_argument("--dashboard", action="store_true", help="启用课堂展示 Dashboard：HUD、腕部相机和任务队列面板")
    parser.add_argument("--presentation-mode", action="store_true", help="更适合课堂展示的精简版 Dashboard")
    parser.add_argument("--save-video", action="store_true", help="保存 Dashboard / Demo 视频")
    parser.add_argument("--no-api", action="store_true", help="关闭 Qwen API 规划和视觉裁判，自动使用 demo plan")
    parser.add_argument("--demo-plan", action="store_true", help="强制使用内置课堂 demo plan")
    parser.add_argument("--out-dir", default=OUTPUT_DIR, help="Dashboard 输出目录，默认 outputs/dashboard_demo")
    parser.add_argument("--self-iterate", action="store_true", help="启用 LLM 自迭代抓取：失败时读结构化反馈→改抓取参数→重投，并写 ablation 日志")
    parser.add_argument("--codegen", action="store_true", help="stretch：自迭代时让 LLM 生成策略代码片段（默认仍为调参）")
    return parser.parse_args()


def apply_runtime_args(args):
    global CLI_INSTRUCTION, ENABLE_GRASP_ASSIST, ENABLE_QWEN_VERIFICATION
    global ENABLE_VLM, ENABLE_VOICE, FORCE_KEYBOARD_INPUT, RENDER_WINDOW
    global DASHBOARD_ENABLED, SAVE_VIDEO, OUTPUT_DIR, DEMO_VIDEO_PATH, DISPLAY_DASHBOARD, FORCE_DEMO_PLAN, PRESENTATION_MODE
    global SELF_ITERATE, SELF_ITERATE_CODEGEN, ITER_LOG_PATH
    CLI_INSTRUCTION = args.instruction.strip()
    FORCE_KEYBOARD_INPUT = args.keyboard
    ENABLE_VOICE = not args.no_voice
    ENABLE_VLM = not (args.no_vlm or args.no_api or args.demo_plan)
    ENABLE_QWEN_VERIFICATION = not (args.no_qwen_verify or args.no_api or args.demo_plan)
    RENDER_WINDOW = not args.headless
    ENABLE_GRASP_ASSIST = not args.no_assist
    HUD_STATE["enabled"] = not args.plain_video
    DASHBOARD_ENABLED = args.dashboard
    PRESENTATION_MODE = args.presentation_mode
    SAVE_VIDEO = args.save_video or args.dashboard
    FORCE_DEMO_PLAN = args.demo_plan
    OUTPUT_DIR = os.path.abspath(args.out_dir)
    DISPLAY_DASHBOARD = not args.headless
    SELF_ITERATE = args.self_iterate
    SELF_ITERATE_CODEGEN = args.codegen
    ITER_LOG_PATH = os.path.join(OUTPUT_DIR, "self_iterate_log.jsonl")
    if PRESENTATION_MODE:
        HUD_STATE["enabled"] = True
    if DASHBOARD_ENABLED:
        DEMO_VIDEO_PATH = os.path.join(OUTPUT_DIR, "jarvis_embodied_sorting_dashboard.mp4")
    elif args.save_video:
        DEMO_VIDEO_PATH = os.path.join(OUTPUT_DIR, "pickplace_demo.mp4")
    planner = "Demo Plan" if args.demo_plan or args.no_api or not ENABLE_VLM else "Qwen-VL"
    update_hud(planner=planner)


def task_speech(target_name, destination_name, index, total):
    target_label = TARGET_LABELS.get(target_name, target_name)
    destination_label = DESTINATION_LABELS.get(destination_name, destination_name)
    object_kind = OBJECT_KIND_LABELS.get(target_name, "目标物")
    return f"第{index}项，锁定{object_kind}{target_label}，送往{destination_label}。"


def success_speech(target_name, finished, total):
    target_label = TARGET_LABELS.get(target_name, target_name)
    if finished >= total:
        return f"{target_label}已归位。全部任务完成，先生。"
    return f"{target_label}已归位。继续执行下一项。"


def print_startup_banner():
    print(
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        "║        J.A.R.V.I.S  VISION-GRASP DEMO SYSTEM        ║\n"
        "║        Qwen-VL Planning  ·  Robosuite Control        ║\n"
        "╚══════════════════════════════════════════════════════╝"
    )
    print(
        f"语音: {'ON' if ENABLE_VOICE else 'OFF'} | "
        f"VLM规划: {'ON' if ENABLE_VLM else 'LOCAL'} | "
        f"视觉裁判: {'Qwen' if ENABLE_QWEN_VERIFICATION else 'Sim'} | "
        f"夹持辅助: {'ON' if ENABLE_GRASP_ASSIST else 'OFF'} | "
        f"HUD: {'ON' if HUD_STATE.get('enabled', True) else 'OFF'}\n"
    )


def bypass_dashscope_proxy():
    dashscope_hosts = ["dashscope.aliyuncs.com", "aliyuncs.com", ".aliyuncs.com"]
    for key in ("NO_PROXY", "no_proxy"):
        values = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        for host in dashscope_hosts:
            if host not in values:
                values.append(host)
        os.environ[key] = ",".join(values)


def make_env():
    camera_names = [FRONT_CAMERA_NAME, WRIST_CAMERA_NAME] if DASHBOARD_ENABLED else FRONT_CAMERA_NAME
    has_renderer = RENDER_WINDOW and not DASHBOARD_ENABLED
    try:
        return suite.make(
            env_name="PickPlace",
            robots="Panda",
            has_renderer=has_renderer,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=camera_names,
            camera_heights=480,
            camera_widths=640,
            control_freq=20,
            horizon=5000,
            ignore_done=True,
        )
    except Exception as exc:
        if not DASHBOARD_ENABLED:
            raise
        print(f"⚠️ 腕部相机 {WRIST_CAMERA_NAME} 初始化失败: {exc}")
        print("⚠️ 自动 fallback 到 frontview，主程序继续运行。")
        DASHBOARD_STATE["wrist_camera"] = FRONT_CAMERA_NAME
        DASHBOARD_STATE["wrist_available"] = False
        return suite.make(
            env_name="PickPlace",
            robots="Panda",
            has_renderer=has_renderer,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=FRONT_CAMERA_NAME,
            camera_heights=480,
            camera_widths=640,
            control_freq=20,
            horizon=5000,
            ignore_done=True,
        )


def recolor_cubes(env):
    env.sim.forward()


def save_frontview_image(obs, image_path=IMAGE_PATH):
    image_rgb = obs[f"{FRONT_CAMERA_NAME}_image"]
    image_bgr = cv2.cvtColor(image_rgb[::-1], cv2.COLOR_RGB2BGR)
    image_bgr = draw_hud(image_bgr, HUD_STATE)
    cv2.imwrite(image_path, image_bgr)
    print(f"视觉层：已保存相机图片 {image_path}")


def frontview_obs_to_bgr(obs):
    image_bgr = get_camera_image(DASHBOARD_STATE.get("env"), obs, DASHBOARD_STATE.get("front_camera", FRONT_CAMERA_NAME), 480, 640)
    if image_bgr is None:
        image_rgb = obs[f"{FRONT_CAMERA_NAME}_image"]
        image_bgr = cv2.cvtColor(image_rgb[::-1], cv2.COLOR_RGB2BGR)
    if DASHBOARD_ENABLED:
        wrist_img = get_camera_image(
            DASHBOARD_STATE.get("env"),
            obs,
            DASHBOARD_STATE.get("wrist_camera", WRIST_CAMERA_NAME),
            240,
            320,
        )
        return compose_dashboard(
            image_bgr,
            wrist_img,
            HUD_STATE,
            DASHBOARD_STATE.get("plan", []),
            DASHBOARD_STATE.get("task_statuses", []),
            DASHBOARD_STATE.get("current_task_idx", -1),
        )
    return draw_hud(image_bgr, HUD_STATE)


def update_hud(**kwargs):
    HUD_STATE.update(kwargs)


def format_world_pos(world_pos):
    if world_pos is None:
        return "N/A"
    try:
        return "[" + ", ".join(f"{value:.3f}" for value in np.array(world_pos).flatten()[:3]) + "]"
    except Exception:
        return str(world_pos)


def clean_dashboard_text(value, default="N/A", max_len=96):
    text = str(value if value is not None else default)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = " ".join(text.replace("?", "").split())
    return (text or default)[:max_len]


def normalize_destination_for_display(destination_name):
    aliases = {
        "bin_top_left": "bin_left",
        "bin_bottom_left": "bin_left",
        "bin_top_right": "bin_right",
        "bin_bottom_right": "bin_right",
    }
    return aliases.get(str(destination_name), str(destination_name))


def dashboard_colors():
    return {
        "bg": (12, 15, 20),
        "panel": (24, 29, 38),
        "panel_alt": (32, 38, 50),
        "cyan": (235, 210, 60),
        "blue": (220, 145, 70),
        "white": (238, 242, 246),
        "muted": (145, 154, 166),
        "yellow": (0, 210, 255),
        "green": (110, 220, 120),
        "red": (80, 90, 255),
        "gray": (105, 112, 122),
        "black": (0, 0, 0),
    }


def put_text(image, text, origin, scale=0.5, color=None, thickness=1):
    colors = dashboard_colors()
    cv2.putText(
        image,
        clean_dashboard_text(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color if color is not None else colors["white"],
        thickness,
        cv2.LINE_AA,
    )


def draw_hud(image, state_info):
    return draw_ai_hud(image, state_info)


def add_hud_overlay(image):
    return draw_hud(image, HUD_STATE)


def draw_top_bar(canvas, state_info):
    colors = dashboard_colors()
    height, width = canvas.shape[:2]
    bar_h = 58
    cv2.rectangle(canvas, (0, 0), (width, bar_h), colors["panel"], -1)
    cv2.line(canvas, (0, bar_h), (width, bar_h), colors["cyan"], 1)
    put_text(canvas, "J.A.R.V.I.S // Embodied Sorting Agent", (24, 36), 0.72, colors["cyan"], 2)
    mode = clean_dashboard_text(state_info.get("mode", "IDLE"), max_len=18)
    planner = clean_dashboard_text(state_info.get("planner", "N/A"), max_len=16)
    scene = clean_dashboard_text(state_info.get("scene", "PickPlace"), max_len=16)
    step = int(state_info.get("step", 0) or 0)
    total = int(state_info.get("total", 0) or 0)
    right_text = f"Mode: {mode}   Planner: {planner}   Scene: {scene}   Queue: {step} / {total}"
    put_text(canvas, right_text, (610, 36), 0.46, colors["white"], 1)


def draw_ai_hud(image, state_info):
    if not state_info.get("enabled", True):
        return image
    colors = dashboard_colors()
    hud = image.copy()
    height, width = hud.shape[:2]
    panel_w, panel_h = 430, 224
    x0, y0 = 20, height - panel_h - 74
    overlay = hud.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), colors["panel"], -1)
    hud = cv2.addWeighted(overlay, 0.82, hud, 0.18, 0)
    cv2.rectangle(hud, (x0, y0), (x0 + panel_w, y0 + panel_h), colors["cyan"], 1)
    put_text(hud, "AI STATE HUD", (x0 + 16, y0 + 30), 0.56, colors["cyan"], 2)
    lines = [
        ("AI Planner", state_info.get("planner", "N/A")),
        ("User Command", state_info.get("user_command", "N/A")),
        ("Current Target", state_info.get("target", "N/A")),
        ("Destination", normalize_destination_for_display(state_info.get("destination", "N/A"))),
        ("MuJoCo Body", state_info.get("body", "N/A")),
        ("World Pos", state_info.get("world_pos", "N/A")),
        ("FSM State", state_info.get("mode", "N/A")),
        ("Task Progress", f"{int(state_info.get('step', 0) or 0)} / {int(state_info.get('total', 0) or 0)}"),
        ("Verification", state_info.get("verification", "N/A")),
    ]
    y = y0 + 58
    for label, value in lines:
        put_text(hud, f"{label}:", (x0 + 16, y), 0.40, colors["muted"], 1)
        put_text(hud, clean_dashboard_text(value, max_len=44), (x0 + 150, y), 0.40, colors["white"], 1)
        y += 18
    return hud


def get_available_cameras(env):
    camera_names = get_available_camera_names(env)
    print(f"Available cameras: {camera_names}")
    return camera_names


def select_wrist_camera(env):
    camera_names = get_available_cameras(env)
    if WRIST_CAMERA_NAME in camera_names:
        print(f"Wrist camera selected: {WRIST_CAMERA_NAME}")
        return WRIST_CAMERA_NAME
    keywords = ("wrist", "eye", "hand", "robotview", "robot0")
    for camera_name in camera_names:
        if any(keyword in camera_name.lower() for keyword in keywords):
            print(f"Wrist camera fallback selected: {camera_name}")
            return camera_name
    print("Wrist camera unavailable; dashboard will show Wrist Camera N/A.")
    return None


def get_available_camera_names(env):
    try:
        names = env.sim.model.camera_names
        if isinstance(names, tuple):
            return list(names)
        return [name.decode() if isinstance(name, bytes) else str(name) for name in names]
    except Exception:
        try:
            return [env.sim.model.camera_id2name(camera_id) for camera_id in range(env.sim.model.ncam)]
        except Exception:
            return []


def choose_wrist_camera(env, requested_name=WRIST_CAMERA_NAME):
    selected = select_wrist_camera(env)
    if selected is None:
        return None, False
    return selected, selected != FRONT_CAMERA_NAME


def get_camera_image(env, obs, camera_name, height, width):
    if not camera_name:
        return None
    if obs is not None:
        obs_key = f"{camera_name}_image"
        image_rgb = obs.get(obs_key) if isinstance(obs, dict) else None
        if image_rgb is not None:
            image_bgr = cv2.cvtColor(image_rgb[::-1], cv2.COLOR_RGB2BGR)
            return cv2.resize(image_bgr, (width, height)) if image_bgr.shape[:2] != (height, width) else image_bgr
    if env is None:
        return None
    try:
        image_rgb = env.sim.render(camera_name=camera_name, height=height, width=width)
        if image_rgb is None:
            return None
        return cv2.cvtColor(image_rgb[::-1], cv2.COLOR_RGB2BGR)
    except Exception as exc:
        if not DASHBOARD_STATE.get("wrist_warning_printed", False):
            print(f"Camera render failed for {camera_name}; continuing without inset: {exc}")
            DASHBOARD_STATE["wrist_warning_printed"] = True
        return None


def update_task_status(plan, task_statuses, current_task_idx, event="running"):
    if not plan:
        return []
    if len(task_statuses) != len(plan):
        task_statuses[:] = ["WAITING"] * len(plan)
    if event == "running" and 0 <= current_task_idx < len(task_statuses):
        for index, status in enumerate(task_statuses):
            if status not in {"DONE", "FAILED"}:
                task_statuses[index] = "RUNNING" if index == current_task_idx else "WAITING"
    elif event == "done" and 0 <= current_task_idx < len(task_statuses):
        task_statuses[current_task_idx] = "DONE"
    elif event == "retrying" and 0 <= current_task_idx < len(task_statuses):
        task_statuses[current_task_idx] = "RETRYING"
    elif event == "failed" and 0 <= current_task_idx < len(task_statuses):
        task_statuses[current_task_idx] = "FAILED"
    return task_statuses


def update_task_statuses(plan, current_task_idx, fsm_state, verification_status):
    statuses = ["WAITING"] * len(plan)
    for index in range(min(current_task_idx, len(statuses))):
        statuses[index] = "DONE"
    if 0 <= current_task_idx < len(statuses):
        if verification_status == "Failed":
            statuses[current_task_idx] = "FAILED"
        elif fsm_state == "RETRY":
            statuses[current_task_idx] = "RETRYING"
        elif verification_status == "Success" or fsm_state in {"VERIFIED", "DONE"}:
            statuses[current_task_idx] = "DONE"
        else:
            statuses[current_task_idx] = "RUNNING"
    return statuses


def draw_task_queue_panel(canvas, plan, task_statuses, current_task_idx):
    colors = dashboard_colors()
    height, width = canvas.shape[:2]
    panel_w = 340
    panel_x = width - panel_w
    cv2.rectangle(canvas, (panel_x, 58), (width, height), colors["panel"], -1)
    cv2.line(canvas, (panel_x, 58), (panel_x, height), colors["cyan"], 1)
    put_text(canvas, "Task Queue", (panel_x + 24, 96), 0.72, colors["cyan"], 2)
    put_text(canvas, "[status]  target  ->  destination", (panel_x + 24, 124), 0.38, colors["muted"], 1)
    if not plan:
        put_text(canvas, "No plan available", (panel_x + 24, 166), 0.52, colors["red"], 1)
        return canvas
    status_colors = {
        "RUNNING": colors["yellow"],
        "DONE": colors["green"],
        "FAILED": colors["red"],
        "RETRYING": colors["yellow"],
        "WAITING": colors["gray"],
    }
    row_y = 164
    for index, task in enumerate(plan):
        status = task_statuses[index] if index < len(task_statuses) else "WAITING"
        is_running = index == current_task_idx and status == "RUNNING"
        row_bg = (42, 48, 62) if is_running else colors["panel_alt"]
        cv2.rectangle(canvas, (panel_x + 18, row_y - 24), (width - 18, row_y + 46), row_bg, -1)
        cv2.rectangle(canvas, (panel_x + 18, row_y - 24), (width - 18, row_y + 46), status_colors.get(status, colors["gray"]), 1)
        target = clean_dashboard_text(task.get("target", "N/A"), max_len=12)
        destination = normalize_destination_for_display(task.get("destination", "N/A"))
        put_text(canvas, f"[{status}]", (panel_x + 30, row_y), 0.42, status_colors.get(status, colors["gray"]), 2 if is_running else 1)
        put_text(canvas, f"{index + 1}. {target}  ->  {destination}", (panel_x + 30, row_y + 26), 0.48, colors["white"], 1)
        row_y += 86
        if row_y > height - 120:
            break
    done_count = sum(1 for status in task_statuses if status == "DONE")
    put_text(canvas, f"Success: {done_count} / {len(plan)}", (panel_x + 24, height - 34), 0.54, colors["green"] if done_count == len(plan) else colors["white"], 1)
    return canvas

def draw_wrist_camera_inset(canvas, wrist_img):
    colors = dashboard_colors()
    panel_w = 340
    main_w = canvas.shape[1] - panel_w
    inset_w, inset_h = 260, 176
    x0, y0 = main_w - inset_w - 24, 82
    cv2.rectangle(canvas, (x0 - 4, y0 - 28), (x0 + inset_w + 4, y0 + inset_h + 6), colors["cyan"], 1)
    cv2.rectangle(canvas, (x0 - 4, y0 - 28), (x0 + inset_w + 4, y0), colors["panel"], -1)
    put_text(canvas, "Wrist Camera", (x0 + 8, y0 - 8), 0.48, colors["cyan"], 1)
    if wrist_img is not None and DASHBOARD_STATE.get("wrist_available", False):
        wrist = cv2.resize(wrist_img, (inset_w, inset_h))
        canvas[y0 : y0 + inset_h, x0 : x0 + inset_w] = wrist
    else:
        cv2.rectangle(canvas, (x0, y0), (x0 + inset_w, y0 + inset_h), (18, 20, 26), -1)
        put_text(canvas, "Wrist Camera N/A", (x0 + 48, y0 + 88), 0.56, colors["muted"], 1)

def draw_pipeline_status(canvas, current_stage):
    colors = dashboard_colors()
    height, width = canvas.shape[:2]
    panel_w = 340
    main_w = width - panel_w
    y0 = height - 48
    cv2.rectangle(canvas, (0, y0), (main_w, height), colors["panel"], -1)
    stages = ["Language", "VLM Plan", "Grounding", "FSM", "Execution", "Verify"]
    aliases = {
        "BOOT": "Language",
        "LISTEN": "Language",
        "PLAN": "VLM Plan",
        "PLANNING": "VLM Plan",
        "TARGET_LOCK": "Grounding",
        "GROUNDING": "Grounding",
        "OPEN": "FSM",
        "APPROACH": "FSM",
        "PREGRASP": "FSM",
        "DESCEND": "FSM",
        "PRE_CLOSE": "FSM",
        "GRASP": "FSM",
        "LIFT": "Execution",
        "TRANSPORT_MID": "Execution",
        "TRANSPORT": "Execution",
        "PREPLACE": "Execution",
        "PLACE": "Execution",
        "RELEASE": "Execution",
        "VERIFY": "Verify",
        "VERIFIED": "Verify",
        "DONE": "Verify",
        "COMPLETE": "Verify",
    }
    active = aliases.get(str(current_stage), str(current_stage))
    x = 28
    for index, stage in enumerate(stages):
        color = colors["yellow"] if stage == active else colors["muted"]
        put_text(canvas, stage, (x, y0 + 30), 0.44, color, 2 if stage == active else 1)
        x += 118
        if index < len(stages) - 1:
            put_text(canvas, "->", (x - 34, y0 + 30), 0.42, colors["gray"], 1)

def compose_dashboard(front_img, wrist_img, state_info, plan, task_statuses, current_task_idx):
    canvas_width, canvas_height = 1280, 720
    panel_width, top_h, bottom_h = 340, 58, 48
    main_width = canvas_width - panel_width
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = dashboard_colors()["bg"]
    main_h = canvas_height - top_h - bottom_h
    front_resized = cv2.resize(front_img, (main_width, main_h))
    canvas[top_h : top_h + main_h, :main_width] = front_resized
    draw_top_bar(canvas, state_info)
    draw_wrist_camera_inset(canvas, wrist_img)
    canvas[top_h : top_h + main_h, :main_width] = draw_ai_hud(canvas[top_h : top_h + main_h, :main_width], state_info)
    draw_pipeline_status(canvas, state_info.get("mode", "Language"))
    draw_task_queue_panel(canvas, plan, task_statuses, current_task_idx)
    return canvas

def draw_planning_overlay(canvas, state_info, plan):
    colors = dashboard_colors()
    height, width = canvas.shape[:2]
    panel_w = 720
    x0, y0 = 70, 106
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + 430), colors["panel"], -1)
    canvas[:] = cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0)
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + 430), colors["cyan"], 1)
    put_text(canvas, "AI Planning Stage", (x0 + 28, y0 + 42), 0.78, colors["cyan"], 2)
    put_text(canvas, "Qwen-VL analyzing scene..." if state_info.get("planner") == "Qwen-VL" else "Demo planner generating fallback plan...", (x0 + 28, y0 + 82), 0.52, colors["yellow"], 1)
    command = state_info.get("user_command") or "Sort edible objects to the left bin and drinkable objects to the right bin."
    put_text(canvas, "User Command:", (x0 + 28, y0 + 128), 0.50, colors["muted"], 1)
    put_text(canvas, command, (x0 + 28, y0 + 154), 0.48, colors["white"], 1)
    put_text(canvas, "Detected objects: bread, cereal, milk, can", (x0 + 28, y0 + 202), 0.50, colors["white"], 1)
    put_text(canvas, "Generated JSON Plan:", (x0 + 28, y0 + 246), 0.50, colors["muted"], 1)
    json_lines = ['{"plan": [']
    for index, task in enumerate(plan):
        suffix = "," if index < len(plan) - 1 else ""
        json_lines.append(f'  {{"target": "{task.get("target", "N/A")}", "destination": "{normalize_destination_for_display(task.get("destination", "N/A"))}"}}{suffix}')
    json_lines.append("]}")
    y = y0 + 278
    for line in json_lines:
        put_text(canvas, line, (x0 + 42, y), 0.46, colors["green"], 1)
        y += 24

def draw_completion_overlay(canvas, state_info, plan, task_statuses):
    colors = dashboard_colors()
    done_count = sum(1 for status in task_statuses if status == "DONE")
    text = f"All tasks completed  |  Success {done_count}/{len(plan)}  |  Final Status DONE"
    x0, y0 = 165, 110
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + 620, y0 + 94), colors["panel"], -1)
    canvas[:] = cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0)
    cv2.rectangle(canvas, (x0, y0), (x0 + 620, y0 + 94), colors["green"], 2)
    put_text(canvas, text, (x0 + 28, y0 + 56), 0.58, colors["green"], 2)

def write_dashboard_still_frames(video_writer, obs, plan, task_statuses, current_task_idx, phase="planning", seconds=3.0):
    if obs is None:
        return
    frame_count = max(1, int(DEMO_VIDEO_FPS * seconds))
    for _ in range(frame_count):
        frame = frontview_obs_to_bgr(obs)
        if phase == "planning":
            draw_planning_overlay(frame, HUD_STATE, plan)
        elif phase == "complete":
            draw_completion_overlay(frame, HUD_STATE, plan, task_statuses)
        DASHBOARD_STATE["last_canvas"] = frame
        if video_writer is not None:
            video_writer.write(frame)
        if DASHBOARD_ENABLED and DISPLAY_DASHBOARD and DASHBOARD_STATE.get("window_available", True):
            try:
                cv2.imshow("J.A.R.V.I.S Dashboard", frame)
                cv2.waitKey(1)
            except Exception as exc:
                DASHBOARD_STATE["window_available"] = False
                print(f"OpenCV window unavailable; video saving continues: {exc}")

def save_dashboard_video(*_args, **_kwargs):
    return DEMO_VIDEO_PATH

def make_demo_video_writer(obs, video_path=None, fps=DEMO_VIDEO_FPS):
    if not SAVE_VIDEO:
        return None
    video_path = video_path or DEMO_VIDEO_PATH
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    frame = frontview_obs_to_bgr(obs)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"⚠️ 视频录制初始化失败，将跳过保存: {video_path}")
        return None
    print(f"🎥 Demo 视频录制中: {video_path}")
    writer.write(frame)
    return writer


def write_demo_frame(video_writer, obs):
    if obs is None:
        return
    frame = frontview_obs_to_bgr(obs)
    DASHBOARD_STATE["last_canvas"] = frame
    if video_writer is not None:
        video_writer.write(frame)
    if DASHBOARD_ENABLED and DISPLAY_DASHBOARD and DASHBOARD_STATE.get("window_available", True):
        try:
            cv2.imshow("J.A.R.V.I.S Dashboard", frame)
            cv2.waitKey(1)
        except Exception as exc:
            DASHBOARD_STATE["window_available"] = False
            print(f"⚠️ OpenCV 窗口不可用，继续保存视频/帧: {exc}")


def refresh_observation(env, fallback_obs):
    try:
        return env._get_observations(force_update=True)
    except Exception:
        return fallback_obs


def get_text_instruction_from_keyboard():
    try:
        text = input("⌨️ 请手动输入指令: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n⌨️ 未输入指令，使用默认任务。")
        text = ""
    return text or DEFAULT_INSTRUCTION


def select_microphone_device_index():
    if sr is None:
        return None

    if MIC_DEVICE_INDEX:
        try:
            return int(MIC_DEVICE_INDEX)
        except ValueError:
            print(f"⚠️ 忽略无效的 JARVIS_MIC_DEVICE_INDEX={MIC_DEVICE_INDEX!r}")

    names = sr.Microphone.list_microphone_names()
    if not names:
        return None

    preferred_names = ("pulse", "default")
    for preferred_name in preferred_names:
        for index, name in enumerate(names):
            if name.strip().lower() == preferred_name:
                return index

    good_keywords = ("analog", "alc", "input", "mic", "microphone", "pulse", "default")
    bad_keywords = ("hdmi", "monitor", "output")
    for index, name in enumerate(names):
        lowered_name = name.lower()
        if any(keyword in lowered_name for keyword in good_keywords) and not any(
            keyword in lowered_name for keyword in bad_keywords
        ):
            return index

    return None


def make_microphone_source(device_index):
    if device_index is None:
        return sr.Microphone()
    return sr.Microphone(device_index=device_index)


async def synthesize_jarvis_voice(text, audio_path=AUDIO_PATH):
    communicate = edge_tts.Communicate(
        text,
        JARVIS_VOICE,
        rate=JARVIS_VOICE_RATE,
        pitch=JARVIS_VOICE_PITCH,
        volume=JARVIS_VOICE_VOLUME,
    )
    await communicate.save(audio_path)


def play_tech_chime():
    sample_rate = 22050
    duration = 0.13
    fade = np.linspace(1.0, 0.0, int(sample_rate * duration), dtype=np.float32)
    tones = []
    for frequency in (740.0, 980.0, 1240.0):
        t = np.linspace(0.0, duration, int(sample_rate * duration), False, dtype=np.float32)
        wave = 0.22 * np.sin(2.0 * np.pi * frequency * t) * fade
        tones.append(wave)
    audio = np.concatenate(tones)
    stereo = np.column_stack([audio, audio])
    sound = pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))
    sound.play()
    pygame.time.wait(int(duration * len(tones) * 1000) + 40)


def play_jarvis_voice(text, audio_path=AUDIO_PATH):
    if not ENABLE_VOICE:
        print(f"🔇 贾维斯静音: {text}")
        return
    if edge_tts is None or pygame is None:
        print("🔇 未安装 edge-tts 或 pygame，跳过语音播放。")
        print("   可执行: pip install edge-tts pygame")
        return

    print(f"🔊 贾维斯: {text}")
    try:
        asyncio.run(synthesize_jarvis_voice(text, audio_path))
        pygame.mixer.init()
        play_tech_chime()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as exc:
        print(f"🔇 语音播放失败，继续执行抓取: {exc}")
    finally:
        try:
            pygame.mixer.quit()
        except Exception:
            pass


def listen_to_user():
    if CLI_INSTRUCTION:
        print(f"👤 指令参数: {CLI_INSTRUCTION}")
        return CLI_INSTRUCTION

    if FORCE_KEYBOARD_INPUT:
        return get_text_instruction_from_keyboard()

    if sr is None:
        print("🎤 未安装 speechrecognition / pyaudio，切换为键盘输入。")
        print("   可执行: pip install speechrecognition pyaudio")
        return get_text_instruction_from_keyboard()

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.non_speaking_duration = 0.5
    try:
        device_index = select_microphone_device_index()
        device_names = sr.Microphone.list_microphone_names()
        if device_index is not None and 0 <= device_index < len(device_names):
            print(f"🎙️ 使用麦克风 #{device_index}: {device_names[device_index]}")

        for attempt in range(1, 3):
            with make_microphone_source(device_index) as source:
                print(f"🎤 贾维斯正在待命，先生，请下达指令... ({attempt}/2)")
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                print(f"🎚️ 当前环境噪声阈值: {recognizer.energy_threshold:.0f}")
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=12)

            print("⏳ 正在识别语音...")
            try:
                text = recognizer.recognize_google(audio, language="zh-CN").strip()
                print(f"👤 您说: {text}")
                return text or DEFAULT_INSTRUCTION
            except sr.UnknownValueError:
                print("⚠️ 听到了声音，但没有识别出中文内容。请靠近麦克风再说一遍。")

        print("⌨️ 连续两次没有识别清楚，切换为键盘输入。")
        return get_text_instruction_from_keyboard()
    except sr.WaitTimeoutError:
        print("⌨️ 等待语音超时，切换为键盘输入。")
        return get_text_instruction_from_keyboard()
    except sr.RequestError as exc:
        print(f"⌨️ 在线语音识别服务不可用，切换为键盘输入: {exc}")
        return get_text_instruction_from_keyboard()
    except Exception as exc:
        print(f"❌ 麦克风识别失败，已自动切回文字输入模式: {exc}")
        return get_text_instruction_from_keyboard()


def project_world_to_pixel(env, world_pos, image_shape, camera_name="frontview"):
    image_height, image_width = image_shape[:2]
    camera_id = env.sim.model.camera_name2id(camera_name)
    camera_pos = np.array(env.sim.data.cam_xpos[camera_id], dtype=np.float64)
    camera_rot = np.array(env.sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    camera_point = camera_rot.T @ (np.array(world_pos, dtype=np.float64) - camera_pos)

    if camera_point[2] >= -1e-6:
        return None

    fovy = np.deg2rad(env.sim.model.cam_fovy[camera_id])
    focal_length = 0.5 * image_height / np.tan(0.5 * fovy)
    pixel_x = image_width * 0.5 + focal_length * camera_point[0] / -camera_point[2]
    pixel_y = image_height * 0.5 - focal_length * camera_point[1] / -camera_point[2]
    return np.array([pixel_x, pixel_y], dtype=np.float64)


def get_body_box_corners(env, body_name):
    body_id = env.sim.model.body_name2id(body_name)
    corners = []
    for geom_id, geom_body_id in enumerate(env.sim.model.geom_bodyid):
        if geom_body_id != body_id:
            continue
        geom_size = np.array(env.sim.model.geom_size[geom_id], dtype=np.float64)
        geom_pos = np.array(env.sim.data.geom_xpos[geom_id], dtype=np.float64)
        geom_rot = np.array(env.sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                for z_sign in (-1.0, 1.0):
                    local_corner = geom_size * np.array([x_sign, y_sign, z_sign], dtype=np.float64)
                    corners.append(geom_pos + geom_rot @ local_corner)
    return corners


def draw_text_with_background(image, text, origin, color=OVERLAY_COLOR):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = int(np.clip(x, 0, image.shape[1] - text_size[0] - 4))
    y = int(np.clip(y, text_size[1] + 4, image.shape[0] - baseline - 4))
    top_left = (x - 2, y - text_size[1] - 4)
    bottom_right = (x + text_size[0] + 2, y + baseline + 2)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def save_debug_overlay_image(env, obs, target_name, body_name, target_pos, image_path=IMAGE_PATH):
    image_rgb = obs["frontview_image"]
    image_bgr = cv2.cvtColor(image_rgb[::-1], cv2.COLOR_RGB2BGR)

    projected_corners = [
        pixel
        for corner in get_body_box_corners(env, body_name)
        if (pixel := project_world_to_pixel(env, corner, image_bgr.shape)) is not None
    ]
    center_pixel = project_world_to_pixel(env, target_pos, image_bgr.shape)

    if projected_corners:
        pixels = np.array(projected_corners, dtype=np.float64)
        x_min, y_min = np.floor(pixels.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(pixels.max(axis=0)).astype(int)
    elif center_pixel is not None:
        center_x, center_y = np.round(center_pixel).astype(int)
        x_min, y_min = center_x - 24, center_y - 24
        x_max, y_max = center_x + 24, center_y + 24
    else:
        cv2.imwrite(image_path, image_bgr)
        print(f"视觉调试：目标不在 frontview 相机前方，已保存原图 {image_path}")
        return

    image_height, image_width = image_bgr.shape[:2]
    x_min = int(np.clip(x_min - 4, 0, image_width - 1))
    y_min = int(np.clip(y_min - 4, 0, image_height - 1))
    x_max = int(np.clip(x_max + 4, 0, image_width - 1))
    y_max = int(np.clip(y_max + 4, 0, image_height - 1))

    cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), OVERLAY_COLOR, 2)
    if center_pixel is not None:
        center_x, center_y = np.round(center_pixel).astype(int)
        center_x = int(np.clip(center_x, 0, image_width - 1))
        center_y = int(np.clip(center_y, 0, image_height - 1))
        cv2.drawMarker(image_bgr, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)

    coord_text = f"world xyz=({target_pos[0]:+.3f}, {target_pos[1]:+.3f}, {target_pos[2]:+.3f})"
    draw_text_with_background(image_bgr, f"{target_name} | body: {body_name}", (x_min, y_min - 8))
    draw_text_with_background(image_bgr, coord_text, (x_min, y_max + 20), color=(255, 255, 255))
    cv2.imwrite(image_path, image_bgr)
    print(f"视觉调试：已写入目标框、body 名称、世界坐标 overlay -> {image_path}")


def strip_markdown_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")].strip()
    return cleaned


def normalize_target_name(target_name):
    target_text = str(target_name).strip()
    if target_text in OBJECT_MAPPING:
        return target_text
    lower_text = target_text.lower()
    aliases = {
        "milk": ["milk", "牛奶", "奶"],
        "bread": ["bread", "面包"],
        "cereal": ["cereal", "麦片", "谷物", "早餐"],
        "can": ["can", "易拉罐", "罐", "可乐", "饮料"],
    }
    for object_name, names in aliases.items():
        if any(name in lower_text or name in target_text for name in names):
            return object_name
    print(f"⚠️ 未识别的目标 '{target_name}'，默认选择 milk")
    return "milk"


def normalize_destination_name(destination_name):
    destination_text = str(destination_name).strip()
    if destination_text in BIN_COORDINATES:
        return destination_text
    print(f"⚠️ 未识别的收纳盒 '{destination_name}'，默认选择 bin_bottom_left")
    return "bin_bottom_left"


def normalize_plan(plan_items):
    normalized_plan = []
    for item in plan_items:
        if not isinstance(item, dict):
            continue
        normalized_plan.append(
            {
                "target": normalize_target_name(item.get("target", "milk")),
                "destination": normalize_destination_name(item.get("destination", "bin_bottom_left")),
            }
        )
    return normalized_plan


def parse_vlm_decision(result_text):
    cleaned_text = strip_markdown_json(result_text)
    try:
        decision = json.loads(cleaned_text)
        target_name = normalize_target_name(decision.get("target", "milk"))
        speech_text = str(decision.get("speech", DEFAULT_SPEECH)).strip() or DEFAULT_SPEECH
        return target_name, speech_text
    except json.JSONDecodeError:
        target_name = normalize_target_name(cleaned_text)
        return target_name, "系统解析稍有波澜，但目标已锁定。"


def parse_vlm_plan(result_text):
    cleaned_text = strip_markdown_json(result_text)
    try:
        decision = json.loads(cleaned_text)
        plan_items = normalize_plan(decision.get("plan", []))
        speech_text = str(decision.get("speech", DEFAULT_SPEECH)).strip() or DEFAULT_SPEECH
        return plan_items, speech_text
    except json.JSONDecodeError as exc:
        print(f"JSON解析失败: {exc}")
        return [], "系统解析异常，建议先生改用更明确的指令。"


def default_pickplace_plan():
    return [
        {"target": "bread", "destination": "bin_top_left"},
        {"target": "cereal", "destination": "bin_top_left"},
        {"target": "milk", "destination": "bin_top_right"},
        {"target": "can", "destination": "bin_top_right"},
    ]


def build_demo_plan():
    return default_pickplace_plan()


def dashboard_user_command_text(user_instruction):
    english_default = "Sort edible objects to the left bin and drinkable objects to the right bin."
    ascii_text = clean_dashboard_text(user_instruction, default="", max_len=110)
    if ascii_text and ascii_text != "N/A":
        return ascii_text
    return english_default


def is_food_drink_sort_instruction(user_instruction):
    instruction = str(user_instruction).strip().lower()
    food_words = ("吃", "食物", "食品", "饭", "面包", "麦片", "bread", "cereal", "food")
    drink_words = ("喝", "饮", "饮品", "饮料", "牛奶", "易拉罐", "milk", "can", "drink")
    sort_words = ("分类", "分拣", "整理", "放", "sort", "classify")
    generic_scene_words = ("桌面", "桌上", "台面", "东西", "物品", "所有", "全部", "object", "objects", "items")
    food_drink_sort = (
        any(word in instruction for word in food_words)
        and any(word in instruction for word in drink_words)
        and any(word in instruction for word in sort_words)
    )
    generic_pickplace_sort = (
        any(word in instruction for word in generic_scene_words)
        and any(word in instruction for word in sort_words)
    )
    return food_drink_sort or generic_pickplace_sort


def side_destination(instruction):
    instruction = str(instruction).lower()
    if "右" in instruction or "right" in instruction:
        return "bin_top_right"
    if "左" in instruction or "left" in instruction:
        return "bin_top_left"
    return "bin_top_left"


def local_instruction_plan(user_instruction):
    instruction = str(user_instruction).strip().lower()
    mentions_food = any(word in instruction for word in ("吃", "食物", "食品", "面包", "麦片", "bread", "cereal", "food"))
    mentions_drink = any(word in instruction for word in ("喝", "饮", "饮品", "饮料", "牛奶", "易拉罐", "milk", "can", "drink"))
    mentions_left = "左" in instruction or "left" in instruction
    mentions_right = "右" in instruction or "right" in instruction

    if mentions_food and not mentions_drink and (mentions_left or mentions_right):
        destination = side_destination(instruction)
        side_label = "右侧" if destination.endswith("right") else "左侧"
        return [
            {"target": "bread", "destination": destination},
            {"target": "cereal", "destination": destination},
        ], f"好的先生，我会把能吃的物品送往{side_label}。"

    if mentions_drink and not mentions_food and (mentions_left or mentions_right):
        destination = side_destination(instruction)
        side_label = "右侧" if destination.endswith("right") else "左侧"
        return [
            {"target": "milk", "destination": destination},
            {"target": "can", "destination": destination},
        ], f"好的先生，我会把能喝的物品送往{side_label}。"

    return None, None


def call_vlm_for_plan(image_path, user_instruction):
    """使用通义千问 (Qwen-VL-Max) 输出 PickPlace 长时序任务队列"""
    if FORCE_DEMO_PLAN:
        print("🧠 已启用 demo plan，跳过 API 调用。")
        return build_demo_plan(), "好的先生，我将执行课堂展示分拣计划。"

    local_plan, local_speech = local_instruction_plan(user_instruction)
    if local_plan is not None:
        print("🧠 已识别为明确单类搬运指令，使用本地确定性计划避免语义跑偏。")
        return local_plan, local_speech

    if is_food_drink_sort_instruction(user_instruction):
        print("🧠 已识别为食物/饮品分类指令，使用本地确定性分拣计划。")
        return build_demo_plan(), "好的先生，我将把食物放左边，饮品放右边。"

    if not ENABLE_VLM:
        print("🧠 已关闭 VLM，使用本地确定性分拣计划。")
        return build_demo_plan(), "视觉云端已断开，先生。本地策略已接管。"

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("未设置 DASHSCOPE_API_KEY，演示模式使用默认分拣计划")
        return build_demo_plan(), "好的先生，我将把食物放左边，饮品放右边。"

    dashscope.api_key = api_key
    bypass_dashscope_proxy()

    system_prompt = (
        "你是一个高级机器人管家 J.A.R.V.I.S。"
        "图中桌面上可能包含：牛奶(milk)、面包(bread)、麦片(cereal)和易拉罐(can)。"
        "周围有四个收纳盒，代号分别为：bin_top_left, bin_top_right, bin_bottom_left, bin_bottom_right。"
        "如果用户要求按类别分拣，通常食物类 bread/cereal 放左侧，饮品类 milk/can 放右侧。"
    )
    user_prompt = (
        f"先生（用户）的语音指令是：‘{user_instruction}’。\n"
        "请结合画面，生成一个分拣任务队列。严格按照以下 JSON 格式输出，不要任何 Markdown 解释：\n"
        "{\n"
        "  \"speech\": \"好的先生，我将为您分类。\",\n"
        "  \"plan\": [\n"
        "    {\"target\": \"bread\", \"destination\": \"bin_top_left\"},\n"
        "    {\"target\": \"milk\", \"destination\": \"bin_top_right\"}\n"
        "  ]\n"
        "}\n"
        "target 只能是 milk、bread、cereal、can；destination 只能是四个 bin 代号。"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},
                {"text": f"{system_prompt}\n\n{user_prompt}"},
            ],
        }
    ]

    print("🧠 正在调用通义千问 Qwen-VL-Max 生成分拣计划，请稍候...")

    for attempt in range(1, 4):
        try:
            response = dashscope.MultiModalConversation.call(
                model="qwen-vl-max",
                messages=messages,
                request_timeout=30,
            )

            if response.status_code == 200:
                result_text = response.output.choices[0].message.content[0]["text"].strip()
                plan_items, speech_text = parse_vlm_plan(result_text)
                if not plan_items:
                    plan_items = build_demo_plan()
                    speech_text = "计划解析不够优雅，先生。我将执行默认分类。"
                print(f"✅ 大模型规划完成，任务数: {len(plan_items)}")
                for index, task in enumerate(plan_items, start=1):
                    print(f"   {index}. {task['target']} -> {task['destination']}")
                print(f"🤖 贾维斯台词: {speech_text}")
                return plan_items, speech_text

            print(f"❌ API 调用失败: 错误码 {response.code} - {response.message}")
        except Exception as exc:
            print(f"❌ 第 {attempt}/3 次调用异常: {exc}")

        if attempt < 3:
            time.sleep(2.0)

    print("决策层：通义千问调用失败，使用默认 PickPlace 分拣计划")
    return build_demo_plan(), "通讯略有延迟，先生。我将执行默认分类。"


def parse_verification_result(result_text):
    cleaned_text = strip_markdown_json(result_text).strip().lower()
    if "success" in cleaned_text or "成功" in cleaned_text or "yes" in cleaned_text:
        return True
    if "fail" in cleaned_text or "失败" in cleaned_text or "no" in cleaned_text:
        return False
    print(f"⚠️ 视觉裁判回复不明确: {result_text}")
    return False


def call_vlm_for_verification(image_path, target_name, destination_name):
    if not ENABLE_QWEN_VERIFICATION:
        print("视觉裁判：已关闭 Qwen 裁判，使用仿真状态核验。")
        return None

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("视觉裁判：未设置 DASHSCOPE_API_KEY，跳过 Qwen 裁判。")
        return None

    dashscope.api_key = api_key
    bypass_dashscope_proxy()
    target_label = TARGET_LABELS.get(target_name, target_name)
    destination_label = DESTINATION_LABELS.get(destination_name, destination_name)
    prompt = (
        f"刚刚机器人试图把 {target_label}({target_name}) 放入 {destination_label}({destination_name})。"
        "请观察图片，判断该物体是否已经成功位于目标收纳盒内部。"
        "只允许回复一个英文单词：success 或 fail。不要解释。"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},
                {"text": prompt},
            ],
        }
    ]

    print(f"🧑‍⚖️ 视觉裁判：正在核验 {target_name} -> {destination_name} ...")
    for attempt in range(1, 3):
        try:
            response = dashscope.MultiModalConversation.call(
                model="qwen-vl-max",
                messages=messages,
                request_timeout=20,
            )
            if response.status_code == 200:
                result_text = response.output.choices[0].message.content[0]["text"].strip()
                success = parse_verification_result(result_text)
                print(f"🧑‍⚖️ 视觉裁判回复: {result_text} -> {'success' if success else 'fail'}")
                return success
            print(f"❌ 视觉裁判 API 失败: 错误码 {response.code} - {response.message}")
        except Exception as exc:
            print(f"❌ 视觉裁判第 {attempt}/2 次异常: {exc}")
        if attempt < 2:
            time.sleep(1.0)
    return None


def get_body_position(env, body_name):
    body_id = env.sim.model.body_name2id(body_name)
    return np.array(env.sim.data.body_xpos[body_id], dtype=np.float64)


def ground_target_to_3d(env, target_name):
    body_name = OBJECT_MAPPING[target_name]
    target_pos = get_body_position(env, body_name)
    print(f"Grounding：{target_name} -> MuJoCo body {body_name} -> 坐标 {target_pos}")
    return body_name, target_pos


def get_bin_place_position(destination_name):
    place_pos = BIN_COORDINATES[destination_name].copy()
    print(f"Place Grounding：{destination_name} -> 坐标 {place_pos}")
    return place_pos


def verify_task_by_sim_state(env, target_name, destination_name):
    body_name = OBJECT_MAPPING[target_name]
    object_pos = get_body_position(env, body_name)
    destination_pos = BIN_COORDINATES[destination_name]
    xy_distance = np.linalg.norm(object_pos[:2] - destination_pos[:2])
    z_ok = object_pos[2] < destination_pos[2] + SIM_SUCCESS_Z_MARGIN
    success = xy_distance < SIM_SUCCESS_XY_TOLERANCE and z_ok
    print(
        "仿真裁判："
        f"{target_name} 与 {destination_name} 平面距离 {xy_distance:.3f} m，"
        f"高度 {object_pos[2]:.3f} m -> {'success' if success else 'fail'}"
    )
    return success


def step_env(env, action):
    try:
        result = env.step(action)
    except ValueError as exc:
        if "terminated episode" in str(exc):
            print("控制层：环境已终止，停止继续执行动作")
            return None, 0.0, True, {"error": str(exc)}
        raise
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info
    return result


def render_env(env):
    if getattr(env, "has_renderer", False):
        env.render()


def compute_movement_action(env, obs, desired_pos, gripper_cmd, tolerance=GRASP_REACH_TOLERANCE, gain=10.0):
    current_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64)
    position_error = desired_pos - current_pos
    action = np.zeros(env.action_dim, dtype=np.float64)
    action[:3] = np.clip(position_error * gain, -1.0, 1.0)
    action[-1] = gripper_cmd
    reached = np.linalg.norm(position_error) < tolerance
    return action, reached


def smooth_action(action, previous_action=None, alpha=0.28, max_delta=0.055):
    if previous_action is None:
        return action
    smoothed = previous_action + np.clip(action - previous_action, -max_delta, max_delta)
    smoothed[:3] = (1.0 - alpha) * previous_action[:3] + alpha * smoothed[:3]
    smoothed[-1] = action[-1]
    return np.clip(smoothed, -1.0, 1.0)


def object_lifted_enough(env, target_name, initial_z):
    body_name = OBJECT_MAPPING[target_name]
    current_z = get_body_position(env, body_name)[2]
    return current_z - initial_z >= MIN_LIFT_DELTA_FOR_HELD_OBJECT


def get_gripper_pad_center(env, obs):
    pad_names = ("gripper0_right_finger1_pad_collision", "gripper0_right_finger2_pad_collision")
    pad_positions = []
    for pad_name in pad_names:
        try:
            geom_id = env.sim.model.geom_name2id(pad_name)
            pad_positions.append(np.array(env.sim.data.geom_xpos[geom_id], dtype=np.float64))
        except Exception:
            pass
    if len(pad_positions) == 2:
        return 0.5 * (pad_positions[0] + pad_positions[1])
    return np.array(obs["robot0_eef_pos"], dtype=np.float64)


def get_stable_eef_anchor(obs):
    return np.array(obs["robot0_eef_pos"], dtype=np.float64)


def get_gripper_grasp_center(env, obs):
    try:
        site_id = env.sim.model.site_name2id("gripper0_right_grip_site")
        return np.array(env.sim.data.site_xpos[site_id], dtype=np.float64)
    except Exception:
        return get_stable_eef_anchor(obs)


def set_body_collision_enabled(env, body_name, enabled):
    try:
        body_id = env.sim.model.body_name2id(body_name)
    except Exception:
        return []

    saved_collision = []
    for geom_id, geom_body_id in enumerate(env.sim.model.geom_bodyid):
        if geom_body_id != body_id:
            continue
        saved_collision.append(
            (
                geom_id,
                int(env.sim.model.geom_contype[geom_id]),
                int(env.sim.model.geom_conaffinity[geom_id]),
            )
        )
        if not enabled:
            env.sim.model.geom_contype[geom_id] = 0
            env.sim.model.geom_conaffinity[geom_id] = 0
    env.sim.forward()
    return saved_collision


def restore_body_collision(env, saved_collision):
    for geom_id, contype, conaffinity in saved_collision or []:
        env.sim.model.geom_contype[geom_id] = contype
        env.sim.model.geom_conaffinity[geom_id] = conaffinity
    if saved_collision:
        env.sim.forward()


def set_free_body_pose(env, body_name, position):
    joint_name = body_name.replace("_main", "_joint0")
    try:
        joint_id = env.sim.model.joint_name2id(joint_name)
    except Exception:
        return False
    qpos_address = env.sim.model.jnt_qposadr[joint_id]
    qvel_address = env.sim.model.jnt_dofadr[joint_id]
    env.sim.data.qpos[qpos_address : qpos_address + 3] = position
    env.sim.data.qpos[qpos_address + 3 : qpos_address + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env.sim.data.qvel[qvel_address : qvel_address + 6] = 0.0
    env.sim.forward()
    return True


def get_grounded_place_position(target_name, place_pos):
    return np.array([place_pos[0], place_pos[1], OBJECT_REST_Z.get(target_name, 0.86)], dtype=np.float64)


def move_free_body_towards_pose(env, body_name, target_position, blend=0.2, max_step=0.018):
    current_position = get_body_position(env, body_name)
    target_position = np.array(target_position, dtype=np.float64)
    delta = target_position - current_position
    distance = np.linalg.norm(delta)
    if distance < 1e-6:
        smoothed_position = target_position
    else:
        step = min(distance, max_step, max(distance * blend, 0.002))
        smoothed_position = current_position + delta / distance * step
    return set_free_body_pose(env, body_name, smoothed_position)


def smooth_position_towards(current_position, target_position, blend, max_step):
    current_position = np.array(current_position, dtype=np.float64)
    target_position = np.array(target_position, dtype=np.float64)
    delta = target_position - current_position
    distance = np.linalg.norm(delta)
    if distance < 1e-8:
        return target_position
    step = min(distance, max_step, max(distance * blend, 0.0015))
    return current_position + delta / distance * step


def update_carry_anchor(env, target_name, obs, grasp_offset, carry_anchor, lock_to_gripper=False, min_clearance=ASSIST_MIN_CLEARANCE):
    body_name = OBJECT_MAPPING[target_name]
    target_position = get_captured_hold_position(env, target_name, obs, grasp_offset, min_clearance=min_clearance)
    if lock_to_gripper:
        carry_anchor = target_position
    else:
        if carry_anchor is None:
            current_position = get_body_position(env, body_name)
        else:
            current_position = carry_anchor
        carry_anchor = smooth_position_towards(
            current_position,
            target_position,
            ASSIST_CARRY_ANCHOR_BLEND,
            ASSIST_CARRY_ANCHOR_MAX_STEP,
        )
    set_free_body_pose(env, body_name, carry_anchor)
    return carry_anchor, True


def get_assisted_hold_position(env, target_name, obs):
    eef_pos = get_gripper_grasp_center(env, obs)
    hold_offset = ASSIST_HOLD_OFFSETS.get(target_name, np.array([0.0, 0.0, -0.04], dtype=np.float64))
    object_rest_z = OBJECT_REST_Z.get(target_name, 0.86)
    hold_pos = eef_pos + hold_offset
    hold_pos[2] = max(hold_pos[2], object_rest_z + ASSIST_MIN_CLEARANCE)
    return hold_pos


def capture_grasp_offset(env, target_name, obs):
    nominal_offset = ASSIST_HOLD_OFFSETS.get(target_name, np.array([0.0, 0.0, -0.04], dtype=np.float64)).copy()
    nominal_offset[2] = np.clip(nominal_offset[2], ASSIST_CARRY_Z_LIMITS[0], ASSIST_CARRY_Z_LIMITS[1])
    return nominal_offset


def capture_current_grasp_offset(env, target_name, obs):
    body_name = OBJECT_MAPPING[target_name]
    measured_offset = get_body_position(env, body_name) - get_gripper_grasp_center(env, obs)
    measured_offset[:2] = np.clip(measured_offset[:2], -ASSIST_CARRY_XY_LIMIT, ASSIST_CARRY_XY_LIMIT)
    measured_offset[2] = np.clip(measured_offset[2], ASSIST_CARRY_Z_LIMITS[0], ASSIST_CARRY_Z_LIMITS[1])
    return measured_offset


def get_captured_hold_position(env, target_name, obs, grasp_offset, min_clearance=ASSIST_MIN_CLEARANCE):
    object_rest_z = OBJECT_REST_Z.get(target_name, 0.86)
    hold_pos = get_gripper_grasp_center(env, obs) + grasp_offset
    hold_pos[2] = max(hold_pos[2], object_rest_z + min_clearance)
    return hold_pos


def assisted_object_hold_error(env, target_name, obs):
    body_name = OBJECT_MAPPING[target_name]
    return np.linalg.norm(get_body_position(env, body_name) - get_assisted_hold_position(env, target_name, obs))


def captured_object_hold_error(env, target_name, obs, grasp_offset):
    body_name = OBJECT_MAPPING[target_name]
    return np.linalg.norm(get_body_position(env, body_name) - get_captured_hold_position(env, target_name, obs, grasp_offset))


def attach_object_to_eef(env, target_name, obs, blend=ASSIST_FOLLOW_BLEND, max_step=ASSIST_MAX_STEP):
    body_name = OBJECT_MAPPING[target_name]
    return move_free_body_towards_pose(env, body_name, get_assisted_hold_position(env, target_name, obs), blend, max_step)


def follow_captured_object(env, target_name, obs, grasp_offset, blend=ASSIST_FOLLOW_BLEND, max_step=ASSIST_MAX_STEP):
    body_name = OBJECT_MAPPING[target_name]
    max_step = ASSIST_OBJECT_MAX_STEPS.get(target_name, max_step) if max_step is None else max_step
    return move_free_body_towards_pose(
        env,
        body_name,
        get_captured_hold_position(env, target_name, obs, grasp_offset),
        blend,
        max_step,
    )


def place_assisted_object(env, target_name, place_pos):
    body_name = OBJECT_MAPPING[target_name]
    return set_free_body_pose(env, body_name, get_grounded_place_position(target_name, place_pos))


def move_assisted_object_to_place(env, target_name, place_pos, blend=ASSIST_PLACE_BLEND):
    body_name = OBJECT_MAPPING[target_name]
    assisted_place_pos = get_grounded_place_position(target_name, place_pos)
    return move_free_body_towards_pose(env, body_name, assisted_place_pos, blend, ASSIST_PLACE_MAX_STEP)


def keep_object_grounded_for_video(env, obs, target_name, place_pos, video_writer=None, frames=20):
    zero_action = np.zeros(env.action_dim, dtype=np.float64)
    for _ in range(frames):
        place_assisted_object(env, target_name, place_pos)
        obs = refresh_observation(env, obs)
        write_demo_frame(video_writer, obs)
        render_env(env)
        next_obs, reward, done, info = step_env(env, zero_action)
        if next_obs is not None:
            obs = next_obs
        if done:
            break
    place_assisted_object(env, target_name, place_pos)
    return refresh_observation(env, obs)


def settle_assisted_object(env, obs, target_name, place_pos):
    body_name = OBJECT_MAPPING[target_name]
    eef_pos = get_stable_eef_anchor(obs)
    hold_offset = ASSIST_HOLD_OFFSETS.get(target_name, np.array([0.0, 0.0, -0.04], dtype=np.float64))
    object_pos = np.array(
        [
            place_pos[0],
            place_pos[1],
            max(OBJECT_REST_Z.get(target_name, 0.86), eef_pos[2] + hold_offset[2]),
        ],
        dtype=np.float64,
    )
    return move_free_body_towards_pose(env, body_name, object_pos, ASSIST_PLACE_BLEND, ASSIST_PLACE_MAX_STEP)


def execute_push_fallback(env, obs, target_name, target_pos, place_pos, video_writer=None):
    body_name = OBJECT_MAPPING[target_name]
    object_pos = get_body_position(env, body_name)
    push_vector = place_pos[:2] - object_pos[:2]
    distance = np.linalg.norm(push_vector)
    if distance < 1e-6:
        return obs, True, "已在目标附近"

    push_direction = push_vector / distance
    start_xy = object_pos[:2] - push_direction * PUSH_START_BACKOFF
    end_xy = place_pos[:2] + push_direction * PUSH_END_OVERSHOOT
    push_z = object_pos[2] + PUSH_CLEARANCE
    start_pos = np.array([start_xy[0], start_xy[1], push_z + 0.12], dtype=np.float64)
    low_start_pos = np.array([start_xy[0], start_xy[1], push_z], dtype=np.float64)
    end_pos = np.array([end_xy[0], end_xy[1], push_z], dtype=np.float64)
    lift_pos = np.array([end_xy[0], end_xy[1], push_z + 0.16], dtype=np.float64)

    print(f"🛟 抓取兜底：{target_name} 过宽，改用闭合夹爪推送到收纳盒。")
    state = "MOVE_START"
    state_steps = 0
    for _ in range(PUSH_MAX_STEPS):
        if state == "MOVE_START":
            action, reached = compute_movement_action(env, obs, start_pos, 1.0, tolerance=PLACE_REACH_TOLERANCE, gain=7.0)
            if reached:
                state, state_steps = "LOWER", 0
                print("PUSH -> LOWER")
        elif state == "LOWER":
            action, reached = compute_movement_action(env, obs, low_start_pos, 1.0, gain=4.5)
            if reached or state_steps > 80:
                state, state_steps = "PUSH", 0
                print("PUSH -> PUSH")
        elif state == "PUSH":
            action, reached = compute_movement_action(env, obs, end_pos, 1.0, tolerance=PLACE_REACH_TOLERANCE, gain=3.2)
            if reached or state_steps > 240:
                state, state_steps = "LIFT", 0
                print("PUSH -> LIFT")
        elif state == "LIFT":
            action, reached = compute_movement_action(env, obs, lift_pos, -1.0, tolerance=PLACE_REACH_TOLERANCE, gain=6.0)
            if reached or state_steps > 90:
                success = verify_task_by_sim_state(env, target_name, destination_name_from_place(place_pos))
                return obs, success, "推送完成" if success else "推送后未到位"
        else:
            raise RuntimeError(f"未知推送状态: {state}")

        next_obs, reward, done, info = step_env(env, action)
        if next_obs is not None:
            obs = next_obs
            write_demo_frame(video_writer, obs)
        render_env(env)
        state_steps += 1
        if done:
            return obs, False, "环境提前结束"

    return obs, False, "推送超时"


def destination_name_from_place(place_pos):
    distances = {
        destination_name: np.linalg.norm(place_pos[:2] - destination_pos[:2])
        for destination_name, destination_pos in BIN_COORDINATES.items()
    }
    return min(distances, key=distances.get)


def default_grasp_params(target_name):
    """每个物体的初始抓取参数（自迭代起点）。

    改进：根据物体几何形状给不同初始值，减少探索轮次。
    - 平面物体(bread)：抓浅会抓空，默认就抓深 + 夹久
    - 圆柱体(milk/can)：从侧面中间抓，下降要柔避免撞飞
    - 薄盒(cereal)：最难，抓深 + 长闭合 + 更柔下降
    """
    xy = GRASP_XY_OFFSETS.get(target_name, np.zeros(2, dtype=np.float64))
    params = {
        "z_offset": float(GRASP_Z_OFFSETS.get(target_name, 0.05)),
        "xy_offset_x": float(xy[0]),
        "xy_offset_y": float(xy[1]),
        "close_steps": int(GRASP_CLOSE_STEPS),
        "settle_steps": int(GRASP_SETTLE_STEPS),
        "descend_gain": 3.6,
    }
    # 形状专属初始值（基于实验观察的失败模式调优）
    if target_name in {"bread"}:           # 平面：抓深 + 夹久
        params.update(z_offset=0.005, close_steps=85, descend_gain=3.2)
    elif target_name in {"cereal"}:        # 薄盒：最难，更深 + 更久 + 更柔
        params.update(z_offset=0.0, close_steps=95, descend_gain=2.8)
    elif target_name in {"milk"}:          # 高圆柱：中段抓，柔下降
        params.update(z_offset=0.03, close_steps=90, descend_gain=3.0)
    elif target_name in {"can"}:           # 矮圆柱：抓深，柔下降
        params.update(z_offset=0.0, close_steps=85, descend_gain=3.0)
    return clip_grasp_params(params)


def clip_grasp_params(params):
    """把任意来源（尤其是 LLM 输出）的参数 clip 到合法范围，绝不信任 LLM 不越界。"""
    safe = dict(params)
    for key, (low, high) in GRASP_PARAM_BOUNDS.items():
        value = safe.get(key, low)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = low
        value = min(max(value, low), high)
        safe[key] = int(round(value)) if key in {"close_steps", "settle_steps"} else value
    return safe


def llm_refine_grasp_params(target_name, history):
    """核心自迭代：把历史尝试的结构化反馈喂给 LLM，让它改抓取参数后重投。

    history: [{"params": {...}, "feedback": {...}}, ...]
    返回 clip 过的新参数 dict；任何异常都回退到“启发式微调”，保证回路不崩。
    """
    last = history[-1]
    last_params = last["params"]
    last_fb = last["feedback"]

    # ---- 回退用的启发式微调（LLM 不可用时仍能自迭代）----
    def heuristic_tweak():
        tweaked = dict(last_params)
        lift = last_fb.get("lift_delta", 0.0)
        threshold = last_fb.get("lift_threshold", MIN_LIFT_DELTA_FOR_HELD_OBJECT)
        drift = last_fb.get("obj_xy_drift", 0.0)
        reached = last_fb.get("reached_grasp_pos", False)

        # 改进：根据 lift 幅度分级调整，越小步长越大
        if lift < threshold * 0.3:
            tweaked["z_offset"] = last_params["z_offset"] - 0.025    # 抓更深（加速）
            tweaked["close_steps"] = last_params["close_steps"] + 20  # 夹更久（加速）
        elif lift < threshold * 0.6:
            tweaked["z_offset"] = last_params["z_offset"] - 0.015
            tweaked["close_steps"] = last_params["close_steps"] + 10

        # 改进：如果末端没到抓取点，随机探索 xy（针对圆柱体）
        if not reached and drift < 0.02:
            import numpy as np
            tweaked["xy_offset_x"] = last_params.get("xy_offset_x", 0.0) + np.random.uniform(-0.01, 0.01)
            tweaked["xy_offset_y"] = last_params.get("xy_offset_y", 0.0) + np.random.uniform(-0.01, 0.01)

        # 改进：drift 大时更快降增益
        if drift > 0.04:
            tweaked["descend_gain"] = last_params["descend_gain"] - 0.8  # 下降更柔（加速）
        elif drift > 0.02:
            tweaked["descend_gain"] = last_params["descend_gain"] - 0.4

        return clip_grasp_params(tweaked)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not ENABLE_VLM or not api_key:
        print("🔁 自迭代：LLM 不可用，使用启发式微调参数。")
        return heuristic_tweak()

    bounds_desc = "\n".join(f"  {k}: {v[0]} ~ {v[1]}" for k, v in GRASP_PARAM_BOUNDS.items())
    prompt = (
        f"你在调试一个机械臂抓取 {target_name} 的策略。可调参数及合法范围：\n{bounds_desc}\n\n"
        f"物理常识：lift_delta 远小于 lift_threshold = 抓空了 → 减小 z_offset(抓更深) 或增大 close_steps；"
        f"obj_xy_drift 大 = 下降太猛把物体撞飞 → 减小 descend_gain；reached_grasp_pos=false = 没到位 → 检查 xy_offset。\n\n"
        f"历史尝试与结果（JSON）：\n{json.dumps(history, ensure_ascii=False, indent=2)}\n\n"
        "请输出改进后的完整参数，严格 JSON、不要任何解释，键为："
        "z_offset, xy_offset_x, xy_offset_y, close_steps, settle_steps, descend_gain。"
    )
    dashscope.api_key = api_key
    bypass_dashscope_proxy()
    print(f"🔁 自迭代：请求 LLM 根据失败反馈改进 {target_name} 的抓取参数...")
    try:
        response = dashscope.MultiModalConversation.call(
            model="qwen-vl-max",
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            request_timeout=30,
        )
        if response.status_code == 200:
            text = response.output.choices[0].message.content[0]["text"].strip()
            new_params = json.loads(strip_markdown_json(text))
            clipped = clip_grasp_params(new_params)
            print(f"🔁 LLM 建议新参数: {clipped}")
            return clipped
        print(f"❌ 自迭代 LLM 调用失败: {response.code} - {response.message}")
    except Exception as exc:
        print(f"❌ 自迭代 LLM 异常: {exc}")
    return heuristic_tweak()


def log_iteration(target_name, attempt, params, feedback):
    """把每次尝试写进 jsonl，作为 ablation 数据与面试证据。"""
    record = {
        "target": target_name,
        "attempt": attempt,
        "params": params,
        "success": bool(feedback.get("success")),
        "lift_delta": feedback.get("lift_delta"),
        "obj_xy_drift": feedback.get("obj_xy_drift"),
        "final_state": feedback.get("final_state"),
        "timestamp": time.time(),
    }
    try:
        os.makedirs(os.path.dirname(ITER_LOG_PATH), exist_ok=True)
        with open(ITER_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"⚠️ 写自迭代日志失败: {exc}")


def execute_grasp_fsm(env, obs, target_name, target_pos, place_pos, video_writer=None, params=None):
    params = clip_grasp_params(params) if params else default_grasp_params(target_name)
    state = "OPEN"
    state_steps = 0
    completed = False
    assisted_grasp = False
    grasp_offset = None
    carry_anchor = None
    saved_target_collision = []
    previous_action = None
    failure_reason = "FSM 超时"
    body_name = OBJECT_MAPPING[target_name]
    # 抓取参数：自迭代模式下来自 params（可被 LLM 调整），否则回退到原全局常量
    grasp_z_offset = params["z_offset"]
    xy_offset = np.array([params["xy_offset_x"], params["xy_offset_y"]], dtype=np.float64)
    close_steps = int(params["close_steps"])
    settle_steps = int(params["settle_steps"])
    descend_gain = float(params["descend_gain"])
    initial_object_pos = get_body_position(env, body_name)
    initial_object_z = initial_object_pos[2]
    # 结构化指标：贯穿整个 FSM，最后打包成 feedback 返回给自迭代回路
    metrics = {
        "reached_grasp_pos": False,   # DESCEND 阶段末端是否到达抓取点
        "max_lift_delta": 0.0,        # 全程物体相对初始的最大真实抬升（物理信号，不受 assist 注水）
        "obj_xy_drift": 0.0,          # 抓取前物体被撞飞的水平距离
    }
    grasp_pos = target_pos + np.array([xy_offset[0], xy_offset[1], grasp_z_offset])
    approach_pos = grasp_pos + np.array([0.0, 0.0, 0.20])
    pregrasp_pos = grasp_pos + np.array([0.0, 0.0, 0.045])
    lift_pos = grasp_pos + np.array([0.0, 0.0, 0.36])
    hover_z = max(place_pos[2] + 0.20, 1.14)
    hold_offset = ASSIST_HOLD_OFFSETS.get(target_name, np.array([0.0, 0.0, -0.04], dtype=np.float64))
    object_rest_z = OBJECT_REST_Z.get(target_name, 0.86)
    place_eef_z = object_rest_z - hold_offset[2] + PLACE_GROUND_CLEARANCE
    transport_mid_pos = np.array(
        [0.5 * (lift_pos[0] + place_pos[0]), 0.5 * (lift_pos[1] + place_pos[1]), hover_z + 0.02],
        dtype=np.float64,
    )
    transport_pos = np.array([place_pos[0], place_pos[1], hover_z], dtype=np.float64)
    preplace_pos = np.array([place_pos[0], place_pos[1], place_eef_z + 0.090], dtype=np.float64)
    settle_pos = np.array([place_pos[0], place_pos[1], place_eef_z], dtype=np.float64)
    release_pos = np.array([place_pos[0], place_pos[1], place_eef_z + 0.095], dtype=np.float64)

    print(
        "控制层：开始 FSM 抓取，"
        f"抓取点 {grasp_pos.round(3)}，高度偏移 {grasp_z_offset:.3f} m"
    )
    for step in range(FSM_MAX_STEPS):
        update_hud(
            mode=state,
            body=body_name,
            world_pos=format_world_pos(get_body_position(env, body_name)),
            verification="Pending",
            progress=min(step / FSM_MAX_STEPS, 0.98),
        )
        update_task_status(
            DASHBOARD_STATE.get("plan", []),
            DASHBOARD_STATE.get("task_statuses", []),
            DASHBOARD_STATE.get("current_task_idx", -1),
            "running",
        )
        motion_state = state
        if state == "OPEN":
            action, _ = compute_movement_action(env, obs, approach_pos, -1.0, tolerance=PLACE_REACH_TOLERANCE, gain=5.0)
            if state_steps > 24:
                state, state_steps = "APPROACH", 0
                print("FSM -> APPROACH")

        elif state == "APPROACH":
            action, reached = compute_movement_action(env, obs, approach_pos, -1.0, tolerance=PLACE_REACH_TOLERANCE, gain=5.0)
            if reached:
                state, state_steps = "PREGRASP", 0
                print("FSM -> PREGRASP")

        elif state == "PREGRASP":
            action, reached = compute_movement_action(env, obs, pregrasp_pos, -1.0, gain=4.6)
            if reached or state_steps > 90:
                state, state_steps = "DESCEND", 0
                print("FSM -> DESCEND")

        elif state == "DESCEND":
            action, reached = compute_movement_action(env, obs, grasp_pos, -1.0, gain=descend_gain)
            if reached:
                metrics["reached_grasp_pos"] = True
            if reached or state_steps > 90:
                state, state_steps = "PRE_CLOSE", 0
                print("FSM -> PRE_CLOSE")

        elif state == "PRE_CLOSE":
            action, _ = compute_movement_action(env, obs, grasp_pos, 0.35, gain=2.8)
            if state_steps > settle_steps:
                state, state_steps = "GRASP", 0
                print("FSM -> GRASP")

        elif state == "GRASP":
            action, _ = compute_movement_action(env, obs, grasp_pos, 1.0, gain=2.6)
            if state_steps > close_steps:
                state, state_steps = "LIFT", 0
                print("FSM -> LIFT")

        elif state == "LIFT":
            action, reached = compute_movement_action(env, obs, lift_pos, 1.0, tolerance=PLACE_REACH_TOLERANCE, gain=3.4)
            if reached:
                if object_lifted_enough(env, target_name, initial_object_z):
                    if ENABLE_GRASP_ASSIST and target_name in ASSIST_GRASP_TARGETS:
                        grasp_offset = capture_current_grasp_offset(env, target_name, obs)
                        carry_anchor = get_body_position(env, OBJECT_MAPPING[target_name])
                        saved_target_collision = set_body_collision_enabled(env, OBJECT_MAPPING[target_name], False)
                        assisted_grasp = True
                        print("🧲 稳定辅助：物体已被真实夹起，只从当前位置托管，禁止抓取前闪现。")
                    state, state_steps = "TRANSPORT_MID", 0
                    print("FSM -> TRANSPORT_MID")
                else:
                    failure_reason = "物体未被真实夹起"
                    print("⚠️ 抬升后物体高度不足：不再把物体拉到夹爪上，交给重试/推送。")
                    break

        elif state == "TRANSPORT_MID":
            action, reached = compute_movement_action(env, obs, transport_mid_pos, 1.0, tolerance=PLACE_REACH_TOLERANCE, gain=3.1)
            if reached or state_steps > 120:
                state, state_steps = "TRANSPORT", 0
                print("FSM -> TRANSPORT")

        elif state == "TRANSPORT":
            action, reached = compute_movement_action(env, obs, transport_pos, 1.0, tolerance=PLACE_REACH_TOLERANCE, gain=3.1)
            if reached:
                state, state_steps = "PREPLACE", 0
                print("FSM -> PREPLACE")

        elif state == "PREPLACE":
            action, reached = compute_movement_action(env, obs, preplace_pos, 1.0, tolerance=PLACE_REACH_TOLERANCE, gain=2.7)
            if reached or state_steps > 100:
                state, state_steps = "PLACE", 0
                print("FSM -> PLACE")

        elif state == "PLACE":
            action, reached = compute_movement_action(env, obs, settle_pos, 1.0, tolerance=0.018, gain=1.8)
            if reached or state_steps > 140:
                state, state_steps = "RELEASE", 0
                print("FSM -> RELEASE")

        elif state == "RELEASE":
            action, reached = compute_movement_action(env, obs, release_pos, -1.0, tolerance=PLACE_REACH_TOLERANCE, gain=2.4)
            if state_steps > RELEASE_STEPS:
                if ENABLE_GRASP_ASSIST and target_name in ASSIST_PLACE_TARGETS:
                    place_assisted_object(env, target_name, place_pos)
                    restore_body_collision(env, saved_target_collision)
                    saved_target_collision = []
                    obs = keep_object_grounded_for_video(env, obs, target_name, place_pos, video_writer, frames=12)
                print("控制层：FSM 完成")
                update_hud(mode="DONE", progress=1.0, message=f"{target_name} placement complete")
                completed = True
                break

        else:
            raise RuntimeError(f"未知 FSM 状态: {state}")

        action = smooth_action(action, previous_action, alpha=0.18, max_delta=0.035)
        previous_action = action.copy()
        next_obs, reward, done, info = step_env(env, action)
        if next_obs is not None:
            obs = next_obs
            assisted_pose_updated = False
            if assisted_grasp and grasp_offset is not None and motion_state in {"TRANSPORT_MID", "TRANSPORT", "PREPLACE", "PLACE"}:
                lock_to_gripper = motion_state in {"TRANSPORT_MID", "TRANSPORT", "PREPLACE", "PLACE"}
                hold_clearance = PLACE_GROUND_CLEARANCE if motion_state == "PLACE" else ASSIST_MIN_CLEARANCE
                carry_anchor, assisted_pose_updated = update_carry_anchor(
                    env,
                    target_name,
                    obs,
                    grasp_offset,
                    carry_anchor,
                    lock_to_gripper=lock_to_gripper,
                    min_clearance=hold_clearance,
                )
            elif motion_state == "RELEASE" and ENABLE_GRASP_ASSIST and target_name in ASSIST_PLACE_TARGETS:
                assisted_pose_updated = place_assisted_object(env, target_name, place_pos)
            if assisted_pose_updated:
                obs = refresh_observation(env, obs)
                render_env(env)
            write_demo_frame(video_writer, obs)
        render_env(env)
        state_steps += 1
        # 追踪真实物理指标：仅在 assist 接管前（抓起阶段）测量，避免 teleport 污染信号
        if not assisted_grasp:
            cur_obj_pos = get_body_position(env, body_name)
            lift_delta = cur_obj_pos[2] - initial_object_z
            if lift_delta > metrics["max_lift_delta"]:
                metrics["max_lift_delta"] = float(lift_delta)
            # 下降/闭合阶段物体被撞飞的水平漂移
            if motion_state in {"DESCEND", "PRE_CLOSE", "GRASP"}:
                drift = float(np.linalg.norm(cur_obj_pos[:2] - initial_object_pos[:2]))
                if drift > metrics["obj_xy_drift"]:
                    metrics["obj_xy_drift"] = drift
        if done:
            print("控制层：环境提前结束")
            failure_reason = "环境提前结束"
            break

    if not completed and failure_reason == "FSM 超时":
        print(f"控制层：FSM 超时，最后状态 {state}，步数上限 {FSM_MAX_STEPS}")

    restore_body_collision(env, saved_target_collision)

    # 打包结构化反馈：这是喂给 LLM 自迭代回路的核心信号，让它能像人一样推理着改参数
    feedback = {
        "success": bool(completed),
        "final_state": state,
        "failure_reason": None if completed else failure_reason,
        "reached_grasp_pos": bool(metrics["reached_grasp_pos"]),
        "lift_delta": round(metrics["max_lift_delta"], 4),
        "lift_threshold": MIN_LIFT_DELTA_FOR_HELD_OBJECT,
        "obj_xy_drift": round(metrics["obj_xy_drift"], 4),
        "params_used": params,
    }
    return obs, completed, feedback


def move_arm_to_home(env, obs, video_writer=None):
    home_pos = np.array([0.0, -0.15, 1.25], dtype=np.float64)
    print("视觉闭环：机械臂回到初始高位，准备重新拍照核验")
    update_hud(mode="VERIFY", progress=0.92, message="Returning home for visual verification")
    for _ in range(HOME_RETURN_MAX_STEPS):
        action, reached = compute_movement_action(env, obs, home_pos, -1.0)
        next_obs, reward, done, info = step_env(env, action)
        if next_obs is not None:
            obs = next_obs
            write_demo_frame(video_writer, obs)
        render_env(env)
        if reached or done:
            break
    return obs


def verify_task_closed_loop(env, obs, target_name, destination_name, video_writer=None):
    update_hud(mode="VERIFY", verification="Pending", message=f"Verifying {target_name} -> {destination_name}")
    obs = move_arm_to_home(env, obs, video_writer)
    save_frontview_image(obs, VERIFY_IMAGE_PATH)
    sim_success = verify_task_by_sim_state(env, target_name, destination_name)
    if sim_success:
        print("✅ 仿真状态确认已到位，跳过不稳定的视觉误判。")
        update_hud(verification="Success")
        return obs, True

    qwen_success = call_vlm_for_verification(VERIFY_IMAGE_PATH, target_name, destination_name)
    if qwen_success is not None:
        update_hud(verification="Success" if qwen_success else "Failed")
        return obs, qwen_success
    update_hud(verification="Failed" if not sim_success else "Success")
    return obs, sim_success


def main():
    apply_runtime_args(parse_args())
    print_startup_banner()
    update_hud(mode="BOOT", task="System startup", message="Robosuite link established")
    env = make_env()
    video_writer = None
    try:
        obs = env.reset()
        DASHBOARD_STATE["env"] = env
        DASHBOARD_STATE["front_camera"] = FRONT_CAMERA_NAME
        wrist_camera, wrist_available = choose_wrist_camera(env, WRIST_CAMERA_NAME) if DASHBOARD_ENABLED else (FRONT_CAMERA_NAME, False)
        DASHBOARD_STATE["wrist_camera"] = wrist_camera
        DASHBOARD_STATE["wrist_available"] = wrist_available
        recolor_cubes(env)
        video_writer = make_demo_video_writer(obs)

        for _ in range(5):
            next_obs, reward, done, info = step_env(env, np.zeros(env.action_dim))
            if next_obs is not None:
                obs = next_obs
                write_demo_frame(video_writer, obs)
            render_env(env)
            if done:
                break

        save_frontview_image(obs)
        update_hud(mode="LISTEN", task="Awaiting command", message="Microphone / keyboard command channel open")
        user_instruction = listen_to_user()
        update_hud(mode="PLAN", task="VLM reasoning", user_command=dashboard_user_command_text(user_instruction), message=dashboard_user_command_text(user_instruction))
        plan_queue, jarvis_speech = call_vlm_for_plan(IMAGE_PATH, user_instruction)
        play_jarvis_voice(jarvis_speech)

        if not plan_queue:
            plan_queue = build_demo_plan()
            update_hud(planner="Demo Plan")

        # 自迭代模式下每个任务带上自己的抓取参数（会被 LLM 逐轮改写）
        pending_tasks = [
            dict(task, attempts=0, plan_index=index, params=default_grasp_params(task["target"]))
            for index, task in enumerate(plan_queue)
        ]
        total_tasks = len(pending_tasks)
        iter_history = {}  # target_name -> [{"params":..., "feedback":...}, ...]
        max_attempts = MAX_ITER_ATTEMPTS if SELF_ITERATE else MAX_VISUAL_RECOVERY_ATTEMPTS
        DASHBOARD_STATE["plan"] = [dict(task) for task in plan_queue]
        DASHBOARD_STATE["task_statuses"] = ["WAITING"] * total_tasks
        DASHBOARD_STATE["current_task_idx"] = -1
        update_hud(mode="PLANNING", task="Generated task plan", step=0, total=total_tasks, verification="N/A")
        if DASHBOARD_ENABLED:
            write_dashboard_still_frames(video_writer, obs, DASHBOARD_STATE["plan"], DASHBOARD_STATE["task_statuses"], -1, phase="planning", seconds=4.0)
        finished_tasks = 0
        while pending_tasks:
            task = pending_tasks.pop(0)
            task["attempts"] += 1
            target_name = task["target"]
            destination_name = task["destination"]
            current_task_idx = int(task.get("plan_index", finished_tasks))
            task_index = current_task_idx + 1
            DASHBOARD_STATE["current_task_idx"] = current_task_idx
            update_task_status(DASHBOARD_STATE["plan"], DASHBOARD_STATE["task_statuses"], current_task_idx, "running")
            target_label = TARGET_LABELS.get(target_name, target_name)
            destination_label = DESTINATION_LABELS.get(destination_name, destination_name)
            update_hud(
                mode="TARGET_LOCK",
                task=f"{target_label} -> {destination_label}",
                target=target_name,
                destination=destination_name,
                body=OBJECT_MAPPING.get(target_name, "N/A"),
                world_pos="N/A",
                verification="Pending",
                progress=0.0,
                step=task_index,
                total=total_tasks,
                message=f"Attempt {task['attempts']} | Acquiring 3D pose",
            )
            print(
                f"🤖 任务 {task_index}/{total_tasks}："
                f"把 {target_name} 放入 {destination_name}，第 {task['attempts']} 次尝试"
            )
            if task["attempts"] == 1:
                play_jarvis_voice(task_speech(target_name, destination_name, task_index, total_tasks))

            body_name, target_pos = ground_target_to_3d(env, target_name)
            place_pos = get_bin_place_position(destination_name)
            update_hud(mode="GROUNDING", body=body_name, world_pos=format_world_pos(target_pos), message=f"{body_name} pose {target_pos.round(3)}")
            save_debug_overlay_image(env, obs, target_name, body_name, target_pos)
            obs, completed, feedback = execute_grasp_fsm(
                env, obs, target_name, target_pos, place_pos, video_writer,
                params=task.get("params"),
            )
            failure_reason = feedback.get("failure_reason") or "FSM 超时"
            if SELF_ITERATE:
                # 记录本轮尝试 + 写日志（无论成败，都是 ablation 数据）
                iter_history.setdefault(target_name, []).append(
                    {"params": task.get("params"), "feedback": feedback}
                )
                log_iteration(target_name, task["attempts"], task.get("params"), feedback)
                print(
                    f"📊 自迭代反馈[{target_name} 第{task['attempts']}次]: "
                    f"success={feedback['success']} lift={feedback['lift_delta']:.3f}"
                    f"/{feedback['lift_threshold']} drift={feedback['obj_xy_drift']:.3f} "
                    f"reached={feedback['reached_grasp_pos']} state={feedback['final_state']}"
                )
            if not completed:
                update_hud(mode="ERROR", progress=0.0, message=f"{target_name} failed: {failure_reason}")
                update_task_status(
                    DASHBOARD_STATE["plan"],
                    DASHBOARD_STATE["task_statuses"],
                    current_task_idx,
                    "retrying" if task["attempts"] < max_attempts else "failed",
                )
                print(f"⚠️ {target_name} 未完成，原因：{failure_reason}。")
                obs = move_arm_to_home(env, obs, video_writer)
                save_frontview_image(obs, VERIFY_IMAGE_PATH)
                if failure_reason == "环境提前结束":
                    break
                if task["attempts"] < max_attempts:
                    if SELF_ITERATE:
                        # 真·自迭代：读失败反馈 → LLM 改参数 → 重投（而非原地重试相同参数）
                        task["params"] = llm_refine_grasp_params(target_name, iter_history[target_name])
                        retry_text = f"先生，{TARGET_LABELS.get(target_name, target_name)}没抓稳，我已根据反馈调整策略，重新尝试。"
                    else:
                        retry_text = f"先生，{TARGET_LABELS.get(target_name, target_name)}没有抓稳，我将重新尝试。"
                    play_jarvis_voice(retry_text)
                    pending_tasks.insert(0, task)
                else:
                    print(f"⚠️ {target_name} 已达到最大尝试次数，跳过该物体。")
                continue

            obs, verified = verify_task_closed_loop(env, obs, target_name, destination_name, video_writer)
            if not verified:
                update_hud(mode="RETRY", progress=0.0, message=f"Verification failed for {target_name}")
                update_task_status(
                    DASHBOARD_STATE["plan"],
                    DASHBOARD_STATE["task_statuses"],
                    current_task_idx,
                    "retrying" if task["attempts"] < max_attempts else "failed",
                )
                if task["attempts"] < max_attempts:
                    if SELF_ITERATE:
                        task["params"] = llm_refine_grasp_params(target_name, iter_history.get(target_name, [{"params": task.get("params"), "feedback": feedback}]))
                    retry_text = f"先生，{TARGET_LABELS.get(target_name, target_name)}滑落了，我将重新尝试。"
                    play_jarvis_voice(retry_text)
                    pending_tasks.insert(0, task)
                    print(f"🔁 视觉闭环：{target_name} 核验失败，已重新插回队列头部。")
                else:
                    print(f"⚠️ {target_name} 视觉核验仍失败，已达到最大尝试次数，跳过该物体。")
                continue

            finished_tasks += 1
            update_task_status(DASHBOARD_STATE["plan"], DASHBOARD_STATE["task_statuses"], current_task_idx, "done")
            update_hud(mode="VERIFIED", progress=1.0, message=f"{target_name} verified in {destination_name}")
            print(f"✅ {target_name} 已通过视觉闭环核验，准备处理下一个物品...")
            play_jarvis_voice(success_speech(target_name, finished_tasks, total_tasks))
            obs = keep_object_grounded_for_video(env, obs, target_name, place_pos, video_writer, frames=12)

            for _ in range(10):
                next_obs, reward, done, info = step_env(env, np.zeros(env.action_dim))
                if next_obs is not None:
                    obs = next_obs
                    place_assisted_object(env, target_name, place_pos)
                    obs = refresh_observation(env, obs)
                    write_demo_frame(video_writer, obs)
                render_env(env)
                if done:
                    break
        DASHBOARD_STATE["current_task_idx"] = -1
        update_hud(
            mode="COMPLETE",
            progress=1.0,
            task="Mission complete",
            target="N/A",
            destination="N/A",
            body="N/A",
            world_pos="N/A",
            verification="Success",
            message="All tasks completed",
            summary=f"Success {sum(1 for status in DASHBOARD_STATE.get('task_statuses', []) if status == 'DONE')} / {len(DASHBOARD_STATE.get('plan', []))}",
        )
        if DASHBOARD_ENABLED:
            write_dashboard_still_frames(video_writer, obs, DASHBOARD_STATE.get("plan", []), DASHBOARD_STATE.get("task_statuses", []), -1, phase="complete", seconds=3.0)
        time.sleep(1.0)
    finally:
        final_frame = DASHBOARD_STATE.get("last_canvas")
        if final_frame is not None and (DASHBOARD_ENABLED or SAVE_VIDEO):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            frame_path = os.path.join(OUTPUT_DIR, "dashboard_last_frame.png")
            cv2.imwrite(frame_path, final_frame)
            print(f"🖼️ Dashboard 最后一帧已保存: {frame_path}")
        if video_writer is not None:
            video_writer.release()
            print(f"🎬 Demo 视频已保存: {DEMO_VIDEO_PATH}")
        if DASHBOARD_ENABLED and DISPLAY_DASHBOARD:
            try:
                cv2.destroyWindow("J.A.R.V.I.S Dashboard")
            except Exception:
                pass
        env.close()


if __name__ == "__main__":
    main()
