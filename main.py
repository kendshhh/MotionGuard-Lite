from __future__ import annotations

import ctypes
import getpass
import hashlib
import hmac
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


PROJECT_TITLE = "MotionGuard Lite: A Python-Based Self-Defense Training Guide and Progress Tracker"
PROGRESS_FILE = Path(__file__).resolve().parent / "progress.txt"
USERS_FILE = Path(__file__).resolve().parent / "users.json"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "graph_opt.pb"
VENV_PYTHON = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
BACK_COMMANDS = {"0", "b", "back", "c", "cancel"}
EXIT_COMMANDS = {"q", "quit", "e", "exit"}
LEGACY_PROGRESS_USERNAME = "legacy"
PROGRESS_HEADER = "timestamp|username|technique|repetitions|successful_reps|accuracy|score|status|detected_technique|recognition_confidence|match_result\n"

ANSI_SUPPORTED: bool | None = None
ANSI_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "white": "\033[37m",
    "gray": "\033[90m",
}
STATUS_COLORS = {
    "passed": "green",
    "needs improvement": "yellow",
    "matched": "green",
    "not matched": "yellow",
    "not run": "gray",
    "correct": "green",
    "adjust": "yellow",
    "pose not clear": "red",
    "camera image too dark": "red",
}

TECHNIQUES = [
    {
        "name": "Block",
        "description": "A defensive move used to protect the face or body from an incoming strike.",
        "steps": [
            "Stand with your feet shoulder-width apart.",
            "Raise both hands near your face to protect your head.",
            "Tuck your elbows close to your body.",
            "Move your forearm outward to intercept the attack.",
        ],
        "situation": "Use this when someone throws a punch or strike toward your upper body.",
    },
    {
        "name": "Punch",
        "description": "A fast straight strike used to create distance and stop an aggressor.",
        "steps": [
            "Keep one hand guarding your face.",
            "Rotate your shoulder and extend your dominant hand forward.",
            "Aim for the center line of the target.",
            "Return your hand quickly to the guard position.",
        ],
        "situation": "Use this when you need to distract an attacker and create space to escape.",
    },
    {
        "name": "Escape",
        "description": "A defensive response for breaking free and moving to safety.",
        "steps": [
            "Protect your head and maintain balance.",
            "Shift your weight and step backward or sideways.",
            "Break contact by pulling toward the attacker's thumb side or weak angle.",
            "Create distance and move immediately to a safe area.",
        ],
        "situation": "Use this when an attacker grabs you and your main goal is to get away safely.",
    },
]


def supports_ansi() -> bool:
    global ANSI_SUPPORTED
    if ANSI_SUPPORTED is not None:
        return ANSI_SUPPORTED

    if not sys.stdout.isatty():
        ANSI_SUPPORTED = False
        return ANSI_SUPPORTED

    if os.name != "nt":
        ANSI_SUPPORTED = True
        return ANSI_SUPPORTED

    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            ANSI_SUPPORTED = False
            return ANSI_SUPPORTED
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
        ANSI_SUPPORTED = True
        return ANSI_SUPPORTED
    except Exception:
        ANSI_SUPPORTED = False
        return ANSI_SUPPORTED


def style_text(text: str, color: str | None = None, bold: bool = False) -> str:
    if not supports_ansi():
        return text

    codes = []
    if bold:
        codes.append(ANSI_CODES["bold"])
    if color:
        codes.append(ANSI_CODES[color])
    if not codes:
        return text
    return f"{''.join(codes)}{text}{ANSI_CODES['reset']}"


def color_for_status(value: str) -> str | None:
    normalized = value.strip().lower()
    for key, color in STATUS_COLORS.items():
        if key in normalized:
            return color
    return None


def status_text(value: str) -> str:
    return style_text(value, color_for_status(value), bold=True)


def print_banner(title: str) -> None:
    line = "=" * len(title)
    print()
    print(style_text(line, "magenta"))
    print(style_text(title, "magenta", bold=True))
    print(style_text(line, "magenta"))


def print_section(title: str, color: str = "cyan") -> None:
    print()
    print(style_text(title, color, bold=True))
    print(style_text("-" * len(title), color))


def print_phase(step: str, title: str, detail: str | None = None) -> None:
    print()
    print(style_text(step, "blue", bold=True))
    print(style_text(title, "white", bold=True))
    if detail:
        print(style_text(detail, "gray"))


def print_kv(label: str, value: str, label_color: str = "cyan", value_color: str | None = None, *, value_bold: bool = False) -> None:
    print(f"{style_text(label + ':', label_color, bold=True)} {style_text(value, value_color, bold=value_bold)}")


def print_status_kv(label: str, value: str) -> None:
    print_kv(label, value, value_color=color_for_status(value), value_bold=True)


def print_info(message: str) -> None:
    print(style_text(message, "gray"))


def print_warning(message: str) -> None:
    print(style_text(message, "yellow", bold=True))


def print_error(message: str) -> None:
    print(style_text(message, "red", bold=True))


def print_success(message: str) -> None:
    print(style_text(message, "green", bold=True))


def show_auth_menu() -> None:
    print_banner(PROJECT_TITLE)
    print(style_text("Authentication", "blue", bold=True))
    print("1. Register")
    print("2. Login")
    print("3. View Leaderboards")
    print("4. Exit Application")


def get_auth_choice() -> int:
    while True:
        user_input = input("Enter your choice (1-4): ").strip()
        if is_exit_command(user_input):
            return 4
        try:
            choice = int(user_input)
        except ValueError:
            print_error("Invalid input. Please enter a number from 1 to 4.")
            continue

        if 1 <= choice <= 4:
            return choice

        print_error("Choice out of range. Please select 1, 2, 3, or 4.")


def show_menu(current_user: str) -> None:
    print_banner(PROJECT_TITLE)
    print(style_text("Main Menu", "blue", bold=True))
    print_kv("Logged In As", current_user, value_color="white", value_bold=True)
    print("1. Start Training")
    print("2. View Techniques Guide")
    print("3. View Progress")
    print("4. View Leaderboards")
    print("5. Logout to Authentication Menu")
    print("6. Exit Application")


def is_back_command(value: str) -> bool:
    return value.strip().lower() in BACK_COMMANDS


def is_exit_command(value: str) -> bool:
    return value.strip().lower() in EXIT_COMMANDS


def wait_for_return(prompt: str = "Press Enter to return to the main menu.") -> None:
    print()
    input(style_text(prompt, "gray"))


def get_menu_choice() -> int:
    while True:
        user_input = input("Enter your choice (1-6): ").strip()
        if is_exit_command(user_input):
            return 6
        try:
            choice = int(user_input)
        except ValueError:
            print_error("Invalid input. Please enter a number from 1 to 6.")
            continue

        if 1 <= choice <= 6:
            return choice

        print_error("Choice out of range. Please select 1, 2, 3, 4, 5, or 6.")


def choose_technique() -> dict | None:
    print_section("Technique Selection", "blue")
    print_info("Choose the number of the technique you want to perform.")
    for index, technique in enumerate(TECHNIQUES, start=1):
        print(f"{index}. {style_text(technique['name'], 'white', bold=True)}")
    print(f"{len(TECHNIQUES) + 1}. Back to Main Menu")

    while True:
        selected = input(f"Choose a technique (1-{len(TECHNIQUES) + 1}): ").strip()
        if is_back_command(selected):
            return None
        if is_exit_command(selected):
            return None
        try:
            technique_index = int(selected)
        except ValueError:
            print_error("Invalid input. Please select one of the numbered options.")
            continue

        if 1 <= technique_index <= len(TECHNIQUES):
            return TECHNIQUES[technique_index - 1]

        if technique_index == len(TECHNIQUES) + 1:
            return None

        print_error("Invalid choice. Please select one of the listed options.")


def view_techniques() -> None:
    print_section("Technique Guide", "blue")
    for technique in TECHNIQUES:
        print()
        print(style_text(technique["name"], "white", bold=True))
        print_kv("Description", technique["description"], value_color="gray")
        print(style_text("Steps", "cyan", bold=True))
        for step_number, step in enumerate(technique["steps"], start=1):
            print(f"  {step_number}. {step}")
        print_kv("Best Situation", technique["situation"], value_color="gray")
    print()
    wait_for_return()


def get_positive_int(prompt: str) -> int | None:
    while True:
        user_input = input(prompt).strip()
        if is_back_command(user_input) or is_exit_command(user_input):
            return None
        try:
            value = int(user_input)
        except ValueError:
            print_error("Invalid input. Please enter a whole number.")
            continue

        if value <= 0:
            print_warning("Value must be greater than zero.")
            continue

        return value


def get_non_negative_int(prompt: str, default: int = 0) -> int | None:
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            return default
        if is_back_command(user_input) or is_exit_command(user_input):
            return None
        try:
            value = int(user_input)
        except ValueError:
            print_error("Invalid input. Please enter a whole number.")
            continue

        if value < 0:
            print_warning("Value cannot be negative.")
            continue

        return value


def get_success_count(repetitions: int) -> int | None:
    while True:
        user_input = input("Enter the number of successful repetitions: ").strip()
        if is_back_command(user_input) or is_exit_command(user_input):
            return None
        try:
            successful_reps = int(user_input)
        except ValueError:
            print_error("Invalid input. Please enter a whole number.")
            continue

        if successful_reps < 0:
            print_warning("Successful repetitions cannot be negative.")
            continue
        if successful_reps > repetitions:
            print_warning("Successful repetitions cannot be greater than total repetitions.")
            continue
        return successful_reps


def ask_yes_no(prompt: str) -> bool | None:
    print()
    print(style_text(prompt, "white", bold=True))
    print("1. Yes")
    print("2. No")
    print("3. Back to Main Menu")
    while True:
        answer = input("Enter your choice (1-3): ").strip().lower()
        if answer in BACK_COMMANDS or answer in EXIT_COMMANDS:
            return None
        if answer in {"1", "y", "yes"}:
            return True
        if answer in {"2", "n", "no"}:
            return False
        if answer == "3":
            return None
        print_error("Please choose 1, 2, or 3.")


def calculate_session_metrics(validation_result: dict | None) -> dict:
    if validation_result is None:
        return {
            "repetitions": 0,
            "successful_reps": 0,
            "accuracy": 0.0,
            "score": 0,
            "status": "Not Run",
            "match_result": "Not Run",
            "match_coverage": 0.0,
            "matched_repetitions": 0,
        }

    matched_repetitions = max(0, int(validation_result.get("matched_repetitions", 0)))
    match_coverage = max(0.0, min(float(validation_result.get("technique_match_ratio", 0.0)), 1.0))
    successful_reps = matched_repetitions
    repetitions = matched_repetitions
    base_score = successful_reps * 10
    coverage_bonus = int(round(match_coverage * 20))
    stance_bonus = 10 if validation_result.get("technique_match") else 0
    score = base_score + coverage_bonus + stance_bonus
    passed = match_coverage >= 0.6 and matched_repetitions >= 1

    return {
        "repetitions": repetitions,
        "successful_reps": successful_reps,
        "accuracy": match_coverage * 100,
        "score": score,
        "status": "Passed" if passed else "Needs Improvement",
        "match_result": "Matched" if validation_result.get("technique_match") else "Not Matched",
        "match_coverage": match_coverage,
        "matched_repetitions": matched_repetitions,
    }


def hash_password(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return derived.hex()


def load_users() -> dict[str, dict[str, str]]:
    if not USERS_FILE.exists():
        return {}
    try:
        payload = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_users(users: dict[str, dict[str, str]]) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2) + "\n", encoding="utf-8")


def prompt_username(prompt: str = "Username: ") -> str | None:
    username = input(prompt).strip()
    if not username or is_back_command(username) or is_exit_command(username):
        return None
    return username


def prompt_password(prompt: str = "Password: ") -> str | None:
    try:
        password = getpass.getpass(prompt)
    except Exception:
        password = input(prompt)
    if not password or is_back_command(password) or is_exit_command(password):
        return None
    return password


def validate_username(username: str) -> str | None:
    if len(username) < 3:
        return "Username must be at least 3 characters long."
    if len(username) > 20:
        return "Username must be 20 characters or fewer."
    if not all(character.isalnum() or character == "_" for character in username):
        return "Username can only contain letters, numbers, and underscores."
    return None


def register_user() -> str | None:
    print_section("Register", "blue")
    print_info("Create an account to access training, progress, and webcam validation.")
    print_info("Password input is hidden while typing. Type your password and press Enter.")
    users = load_users()

    while True:
        username = prompt_username()
        if username is None:
            return None
        username_error = validate_username(username)
        if username_error:
            print_error(username_error)
            continue
        if username in users:
            print_error("That username is already registered.")
            continue

        password = prompt_password()
        if password is None:
            return None
        if len(password) < 4:
            print_error("Password must be at least 4 characters long.")
            continue

        confirm_password = prompt_password("Confirm password: ")
        if confirm_password is None:
            return None
        if not hmac.compare_digest(password, confirm_password):
            print_error("Passwords do not match.")
            continue

        salt_hex = os.urandom(16).hex()
        users[username] = {
            "salt": salt_hex,
            "password_hash": hash_password(password, salt_hex),
        }
        save_users(users)
        print_success(f"Registration successful. Logged in as {username}.")
        return username


def login_user() -> str | None:
    print_section("Login", "blue")
    print_info("Password input is hidden while typing. Type your password and press Enter.")
    users = load_users()
    if not users:
        print_warning("No registered users were found. Create an account first.")
        return None

    username = prompt_username()
    if username is None:
        return None
    password = prompt_password()
    if password is None:
        return None

    stored = users.get(username)
    if not stored:
        print_error("Account not found.")
        return None

    password_hash = hash_password(password, stored.get("salt", ""))
    if not hmac.compare_digest(password_hash, stored.get("password_hash", "")):
        print_error("Incorrect password.")
        return None

    print_success(f"Login successful. Welcome back, {username}.")
    return username


def authenticate_user() -> str | None:
    while True:
        show_auth_menu()
        choice = get_auth_choice()
        if choice == 1:
            username = register_user()
            if username is not None:
                return username
        elif choice == 2:
            username = login_user()
            if username is not None:
                return username
        elif choice == 3:
            view_leaderboards()
        else:
            return None


def is_missing_cv2_error(error: ModuleNotFoundError) -> bool:
    return error.name == "cv2" or "cv2" in str(error).lower() or "opencv" in str(error).lower()


def probe_cameras_with_venv() -> list[dict]:
    if not VENV_PYTHON.is_file():
        return []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_path = Path(temp_file.name)

    try:
        command = [
            str(VENV_PYTHON),
            str(Path(__file__).resolve().parent / "openpose.py"),
            "--list-cameras-json",
            str(temp_path),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0 or not temp_path.exists():
            return []
        payload = json.loads(temp_path.read_text(encoding="utf-8"))
        cameras = payload.get("cameras", [])
        return cameras if isinstance(cameras, list) else []
    except Exception:
        return []
    finally:
        temp_path.unlink(missing_ok=True)


def ensure_progress_file_format() -> None:
    if not PROGRESS_FILE.exists():
        PROGRESS_FILE.write_text(PROGRESS_HEADER, encoding="utf-8")
        return

    try:
        lines = PROGRESS_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    if not lines:
        PROGRESS_FILE.write_text(PROGRESS_HEADER, encoding="utf-8")
        return

    if lines[0] == PROGRESS_HEADER.strip():
        return

    legacy_header = "timestamp|technique|repetitions|successful_reps|accuracy|score|status|detected_technique|recognition_confidence|match_result"
    if lines[0] != legacy_header:
        return

    migrated_lines = [PROGRESS_HEADER.strip()]
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("|")
        if len(parts) == 7:
            migrated_lines.append(
                f"{parts[0]}|{LEGACY_PROGRESS_USERNAME}|{parts[1]}|{parts[2]}|{parts[3]}|{parts[4]}|{parts[5]}|{parts[6]}|Not Run|0.00|Not Run"
            )
        elif len(parts) >= 10:
            migrated_lines.append(
                f"{parts[0]}|{LEGACY_PROGRESS_USERNAME}|{parts[1]}|{parts[2]}|{parts[3]}|{parts[4]}|{parts[5]}|{parts[6]}|{parts[7]}|{parts[8]}|{parts[9]}"
            )

    PROGRESS_FILE.write_text("\n".join(migrated_lines) + "\n", encoding="utf-8")


def get_detected_cameras() -> list[dict]:
    cameras = probe_cameras_with_venv()
    if cameras:
        return cameras

    try:
        from openpose import discover_working_cameras

        return discover_working_cameras()
    except Exception:
        return []


def format_detected_cameras(cameras: list[dict]) -> str:
    formatted = []
    for camera in cameras:
        index = camera.get("index")
        backend = camera.get("backend") or "Auto"
        formatted.append(f"{index} ({backend})")
    return ", ".join(formatted)


def run_webcam_validation_with_venv(
    camera_index: int,
    validation_frames: int | None,
    target_technique: str,
    confirm_frames: int = 5,
) -> dict:
    if not VENV_PYTHON.is_file():
        raise FileNotFoundError("Project virtual environment Python was not found at .venv\\Scripts\\python.exe")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_path = Path(temp_file.name)

    try:
        command = [
            str(VENV_PYTHON),
            str(Path(__file__).resolve().parent / "openpose.py"),
            "--camera-index",
            str(camera_index),
            "--target-technique",
            target_technique,
            "--confirm-frames",
            str(confirm_frames),
            "--json-output",
            str(temp_path),
        ]
        if validation_frames is not None:
            command.extend(["--max-frames", str(validation_frames)])
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError("OpenCV webcam validation could not complete in the project virtual environment.")

        return json.loads(temp_path.read_text(encoding="utf-8"))
    finally:
        temp_path.unlink(missing_ok=True)


def save_progress(record: dict, username: str) -> None:
    ensure_progress_file_format()

    line = (
        f"{record['timestamp']}|{username}|{record['technique']}|{record['repetitions']}|"
        f"{record['successful_reps']}|{record['accuracy']:.2f}|{record['score']}|{record['status']}|"
        f"{record['detected_technique']}|{record['recognition_confidence']:.2f}|{record['match_result']}\n"
    )
    with PROGRESS_FILE.open("a", encoding="utf-8") as progress_file:
        progress_file.write(line)


def parse_progress_record(parts: list[str]) -> dict | None:
    if len(parts) == 7:
        return {
            "timestamp": parts[0],
            "username": LEGACY_PROGRESS_USERNAME,
            "technique": parts[1],
            "repetitions": int(parts[2]),
            "successful_reps": int(parts[3]),
            "accuracy": float(parts[4]),
            "score": int(parts[5]),
            "status": parts[6],
            "detected_technique": "Not Run",
            "recognition_confidence": 0.0,
            "match_result": "Not Run",
        }
    if len(parts) >= 11:
        return {
            "timestamp": parts[0],
            "username": parts[1],
            "technique": parts[2],
            "repetitions": int(parts[3]),
            "successful_reps": int(parts[4]),
            "accuracy": float(parts[5]),
            "score": int(parts[6]),
            "status": parts[7],
            "detected_technique": parts[8],
            "recognition_confidence": float(parts[9]),
            "match_result": parts[10],
        }
    if len(parts) >= 10:
        return {
            "timestamp": parts[0],
            "username": LEGACY_PROGRESS_USERNAME,
            "technique": parts[1],
            "repetitions": int(parts[2]),
            "successful_reps": int(parts[3]),
            "accuracy": float(parts[4]),
            "score": int(parts[5]),
            "status": parts[6],
            "detected_technique": parts[7],
            "recognition_confidence": float(parts[8]),
            "match_result": parts[9],
        }
    return None


def load_all_progress() -> list[dict]:
    ensure_progress_file_format()
    if not PROGRESS_FILE.exists():
        return []

    records: list[dict] = []
    try:
        with PROGRESS_FILE.open("r", encoding="utf-8") as progress_file:
            lines = progress_file.readlines()
    except OSError:
        return []

    for line in lines[1:]:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        parts = stripped_line.split("|")
        record = parse_progress_record(parts)
        if record is not None:
            records.append(record)
    return records


def load_progress(username: str) -> list[dict]:
    return [record for record in load_all_progress() if record.get("username") == username]


def build_leaderboard_entries() -> list[dict]:
    leaderboard: dict[str, dict[str, float | int | str]] = {}
    for record in load_all_progress():
        username = str(record.get("username", LEGACY_PROGRESS_USERNAME))
        if username not in leaderboard:
            leaderboard[username] = {
                "username": username,
                "sessions": 0,
                "total_score": 0,
                "best_score": 0,
                "passed_sessions": 0,
                "total_accuracy": 0.0,
            }
        entry = leaderboard[username]
        score = int(record.get("score", 0))
        accuracy = float(record.get("accuracy", 0.0))
        entry["sessions"] += 1
        entry["total_score"] += score
        entry["best_score"] = max(int(entry["best_score"]), score)
        entry["total_accuracy"] += accuracy
        if record.get("status") == "Passed":
            entry["passed_sessions"] += 1

    ranked_entries: list[dict] = []
    for entry in leaderboard.values():
        sessions = max(1, int(entry["sessions"]))
        success_rate = (int(entry["passed_sessions"]) / sessions) * 100
        average_accuracy = float(entry["total_accuracy"]) / sessions
        ranked_entries.append(
            {
                "username": entry["username"],
                "sessions": int(entry["sessions"]),
                "total_score": int(entry["total_score"]),
                "best_score": int(entry["best_score"]),
                "success_rate": success_rate,
                "average_accuracy": average_accuracy,
            }
        )

    ranked_entries.sort(
        key=lambda item: (
            -item["total_score"],
            -item["best_score"],
            -item["success_rate"],
            item["username"],
        )
    )
    return ranked_entries


def view_leaderboards(current_user: str | None = None) -> None:
    print_section("Leaderboards", "blue")
    if current_user is not None:
        print_kv("Viewing As", current_user, value_color="white", value_bold=True)

    entries = build_leaderboard_entries()
    if not entries:
        print_warning("No leaderboard data is available yet.")
        print()
        wait_for_return("Press Enter to return.")
        return

    for position, entry in enumerate(entries[:10], start=1):
        print()
        title = f"#{position} | {entry['username']}"
        if current_user is not None and entry["username"] == current_user:
            title += " (You)"
        print(style_text(title, "white", bold=True))
        print_kv("Total Score", str(entry["total_score"]), value_color="white", value_bold=True)
        print_kv("Best Score", str(entry["best_score"]), value_color="white", value_bold=True)
        print_kv("Sessions", str(entry["sessions"]), value_color="white", value_bold=True)
        print_kv("Success Rate", f"{entry['success_rate']:.2f}%", value_color="white", value_bold=True)
        print_kv("Average Accuracy", f"{entry['average_accuracy']:.2f}%", value_color="white", value_bold=True)

    print()
    wait_for_return("Press Enter to return.")


def start_training(current_user: str) -> None:
    print_section("Training Mode", "blue")
    print_phase("PHASE 1", "Technique Choice", "Select the move you want to practice.")
    technique = choose_technique()
    if technique is None:
        print_warning("Training canceled. Returning to the main menu.")
        return

    print_section("Selected Technique")
    print_kv("Technique", technique["name"], value_color="white", value_bold=True)
    print_kv("Description", technique["description"], value_color="gray")

    validation_result = None
    print_phase("PHASE 2", "Validation Setup", "Choose whether to use the webcam for live guidance.")
    use_webcam_validation = ask_yes_no("Use OpenCV webcam validation for this technique?")
    if use_webcam_validation is None:
        print_warning("Training canceled. Returning to the main menu.")
        return
    if use_webcam_validation:
        print_phase("PHASE 2A", "Camera Setup", "The app will use the default webcam for live validation.")
        detected_cameras = get_detected_cameras()
        default_camera_index = 0
        if detected_cameras:
            default_camera_index = int(detected_cameras[0].get("index", 0))
            print_info(f"Detected camera indices: {format_detected_cameras(detected_cameras)}")
        camera_index = default_camera_index
        print_info(f"Using webcam index {camera_index}.")

        validation_frames = None
        print_info("The OpenCV webcam window will open with on-screen feedback for the selected move.")
        print_info("Press F for fullscreen, or Q / Esc when you want to finish webcam validation.")
        try:
            from openpose import analyze_stream_source

            validation_result = analyze_stream_source(
                input_source=camera_index,
                model_path=MODEL_PATH,
                display=True,
                max_frames=validation_frames,
                target_technique=technique["name"],
                confirm_frames=5,
            )
        except ModuleNotFoundError as exc:
            if not is_missing_cv2_error(exc):
                print_error(f"Webcam validation could not run: {exc}")
                validation_result = None
            else:
                print_warning("OpenCV is missing in the active interpreter. Retrying webcam validation with the project virtual environment.")
                try:
                    validation_result = run_webcam_validation_with_venv(
                        camera_index=camera_index,
                        validation_frames=validation_frames,
                        target_technique=technique["name"],
                    )
                except Exception as fallback_exc:
                    print_error(f"Webcam validation could not run: {fallback_exc}")
                    validation_result = None
        except Exception as exc:
            print_error(f"Webcam validation could not run: {exc}")
            validation_result = None

    metrics = calculate_session_metrics(validation_result)
    repetitions = metrics["repetitions"]
    successful_reps = metrics["successful_reps"]
    accuracy = metrics["accuracy"]
    score = metrics["score"]
    status = metrics["status"]
    detected_technique = technique["name"]
    openpose_detection = validation_result["recognized_technique"] if validation_result is not None else "Not Run"
    recognition_confidence = validation_result["recognized_confidence"] * 100 if validation_result is not None else 0.0
    match_result = metrics["match_result"]

    print_phase("PHASE 3", "Session Summary", "Review the outcome for this practice session.")
    print_section("Session Summary")
    print_kv("Technique", technique["name"], value_color="white", value_bold=True)
    if validation_result is not None:
        print_kv("Camera Source", validation_result.get("input_source", f"webcam {camera_index}"), value_color="white", value_bold=True)
        print_kv("Matched Repetitions", str(metrics["matched_repetitions"]), value_color="white", value_bold=True)
        print_kv("Match Coverage", f"{metrics['match_coverage'] * 100:.2f}%", value_color="white", value_bold=True)
    else:
        print_kv("Camera Validation", "Not Run", value_color="gray", value_bold=True)
    print_kv("Accuracy", f"{accuracy:.2f}%", value_color="white", value_bold=True)
    print_kv("Score", str(score), value_color="white", value_bold=True)
    print_status_kv("Session Status", status)
    print_status_kv("Match Result", match_result)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "technique": technique["name"],
        "repetitions": repetitions,
        "successful_reps": successful_reps,
        "accuracy": accuracy,
        "score": score,
        "status": status,
        "detected_technique": openpose_detection,
        "recognition_confidence": recognition_confidence,
        "match_result": match_result,
    }
    save_progress(record, current_user)
    print()
    print_success("Progress saved successfully.")


def view_progress(current_user: str) -> None:
    print_section("Training Progress", "blue")
    print_kv("Account", current_user, value_color="white", value_bold=True)
    records = load_progress(current_user)

    if not records:
        print_warning("No progress records found for this account yet.")
        print()
        wait_for_return()
        return

    total_score = 0
    total_sessions = len(records)
    passed_sessions = 0

    for index, record in enumerate(records, start=1):
        print()
        print(style_text(f"Session {index} | {record['timestamp']}", "white", bold=True))
        print_kv("Technique", record["technique"], value_color="white", value_bold=True)
        print_kv("Matched Repetitions", str(record["successful_reps"]), value_color="white", value_bold=True)
        print_kv("Accuracy", f"{record['accuracy']:.2f}%", value_color="white", value_bold=True)
        print_kv("Score", str(record["score"]), value_color="white", value_bold=True)
        print_status_kv("Session Status", record["status"])
        print_status_kv("Match Result", record["match_result"])
        total_score += record["score"]
        if record["status"] == "Passed":
            passed_sessions += 1

    average_score = total_score / total_sessions
    success_rate = (passed_sessions / total_sessions) * 100

    print_section("Progress Summary")
    print_kv("Total Sessions", str(total_sessions), value_color="white", value_bold=True)
    print_kv("Total Progress Score", str(total_score), value_color="white", value_bold=True)
    print_kv("Average Score", f"{average_score:.2f}", value_color="white", value_bold=True)
    print_kv("Successful Sessions", str(passed_sessions), value_color="white", value_bold=True)
    success_rate_color = "green" if success_rate >= 70 else "yellow" if success_rate >= 40 else "red"
    print_kv("Success Rate", f"{success_rate:.2f}%", value_color=success_rate_color, value_bold=True)
    print()
    wait_for_return()


def main() -> None:
    print(style_text(f"Welcome to {PROJECT_TITLE}", "white", bold=True))
    while True:
        current_user = authenticate_user()
        if current_user is None:
            print()
            print_success("Thank you for using MotionGuard Lite.")
            break

        while True:
            show_menu(current_user)
            choice = get_menu_choice()

            if choice == 1:
                start_training(current_user)
            elif choice == 2:
                view_techniques()
            elif choice == 3:
                view_progress(current_user)
            elif choice == 4:
                view_leaderboards(current_user)
            elif choice == 5:
                print()
                print_success(f"Logged out from {current_user}.")
                break
            else:
                print()
                print_success("Thank you for using MotionGuard Lite.")
                return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_warning("Program interrupted by user.")