# MotionGuard Lite

MotionGuard Lite: A Python-Based Self-Defense Training Guide and Progress Tracker is a terminal application built with Python. It helps users explore self-defense techniques, simulate training sessions, and track progress using a simple text-based record system.

## Main Features

- Menu-driven terminal interface
- Register, login, leaderboard viewing, logout, and exit flow before and after accessing training features
- Self-defense technique guide
- Training mode with webcam validation or manual summary entry, plus automatic score calculation
- Per-user progress tracking using local account storage and `progress.txt`
- Phased terminal flow with grouped sections, spacing, and highlighted key results
- Color-coded status feedback in the terminal and a dashboard-style OpenCV webcam window
- Input validation and error handling

## Project Structure

- `main.py` - main terminal application
- `openpose.py` - OpenCV pose validation and on-screen feedback
- `models/graph_opt.pb` - OpenPose model file used by webcam validation
- `progress.txt` - saved training history, now tagged per user account
- `users.json` - generated automatically to store registered local accounts
- `PROJECT_DOCUMENTATION.md` - course-ready project documentation file
- `PRESENTATION_SLIDES.md` - presentation content outline for the final demo
- `PRESENTATION_SLIDES.html` - presentation slide file for the final demo
- `REQUIREMENTS_CHECKLIST.md` - requirement-to-system compliance checklist
- `run_main.bat` - launches the terminal app with the project virtual environment
- `run_webcam.bat` - launches webcam pose validation with the project virtual environment
- `install_requirements.bat` - installs Python dependencies into the project virtual environment
- `requirements.txt` - Python dependencies for the current system
- `reference_poses/` - required reference images for `block`, `punch`, and `escape`

## How To Run

Prefer the project virtual environment so the OpenCV dependency is available:

```powershell
.\.venv\Scripts\python.exe main.py
```

You can also use the included batch files:

```powershell
.\run_main.bat
.\run_webcam.bat --target-technique Punch
.\run_webcam.bat --preset fast --target-technique Punch
.\run_webcam.bat --camera-index 1 --target-technique Punch
```

If dependencies are missing, install them into the virtual environment:

```powershell
.\install_requirements.bat
```

The OpenPose CLI command must be entered once. For example:

```powershell
.\.venv\Scripts\python.exe openpose.py --target-technique Punch
.\.venv\Scripts\python.exe openpose.py --preset fast --target-technique Punch
.\.venv\Scripts\python.exe openpose.py --camera-index 1 --target-technique Punch
```

While the OpenCV webcam window is open, press `F` to toggle fullscreen and press `Q` or `Esc` to close the webcam session.

Before webcam validation can run, add reference images to `reference_poses/`.
The app accepts `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, or `.ppm` files whose names contain the technique name, for example:

```text
reference_poses/punch.jpg
reference_poses/real_escape_pose.png
reference_poses/block_reference.jpeg
reference_poses/punch.ppm
reference_poses/escape.ppm
```

If you run `python main.py` with a different interpreter, you may see `ModuleNotFoundError: No module named 'cv2'` even when the project venv is set up correctly.
If Windows opens the wrong webcam, use `--camera-index 1` or another non-zero index to try a different device.
The training flow now probes the project venv for working camera indices before webcam validation starts, and failed webcam opens report any other detected indices from the probe.
Webcam validation now supports presets: `--preset fast`, `--preset balanced`, and `--preset strict`. `fast` is the recommended default for quicker punch, block, and escape recognition, and it can count close poses as a `Near Match` instead of requiring a perfect pose.
For `Punch`, the detector now expects one hand to stay near your face while the other hand extends at about shoulder height through the center line.
For `Escape`, the detector now expects you to keep at least one hand protecting your head while you move backward or sideways far enough to shift your body off the center line.

## Menu Options

Authentication Menu:
1. Register
2. Login
3. View Leaderboards
4. Exit Application

Main Menu After Login:
1. Start Training
2. View Techniques Guide
3. View Progress
4. View Leaderboards
5. Logout to Authentication Menu
6. Exit Application

## Training Flow

1. Register a new account or log in with an existing one before entering the main menu.
2. Leaderboards can be viewed from the authentication menu before logging in.
3. After login, choose a menu option from 1 to 6.
4. If you choose `1. Start Training`, select the technique number you want to perform, or choose the numbered back option.
5. When asked to use OpenCV webcam validation, choose `1` for Yes, `2` for No, or `3` to go back to the main menu.
6. Training uses phased terminal sections: `PHASE 1` for technique choice, `PHASE 2` for validation setup, conditional `PHASE 2A` for camera setup or manual summary entry, and `PHASE 3` for the final session summary.
7. If you choose webcam validation, the app automatically uses the default detected webcam.
8. A reference image for the selected technique must exist in `reference_poses/`, or training is canceled and returns to the menu.
9. If a webcam is detected during the probe, the app shows the detected indices and uses the first detected device as the default camera index.
10. After the camera is selected, choose a webcam preset: `Fast` for quicker response and near-match support, `Balanced` for a middle ground, or `Strict` for slower but steadier matching.
11. The OpenCV webcam window shows the live camera feed side-by-side with the selected technique reference image, and the reference image is resized/cropped automatically to fit the window.
12. For `Punch`, keep one hand guarding your face and extend the other hand through the center line at shoulder height; hold it briefly so the detector can confirm the pose.
13. For `Escape`, keep at least one hand protecting your head and move backward or sideways enough for your shoulders and hips to shift off the center line.
14. In `Fast`, the OpenCV window can show `Near Match` when the pose is close enough to count even if it is not fully aligned yet. `Balanced` and `Strict` still require tighter alignment.
15. If you choose `2. No`, the app skips the webcam window and asks you to enter manual repetitions and successful repetitions for the session summary.
16. The final terminal summary is generated from either webcam validation or manual entry and shows the technique, score, accuracy, session status, and match result.
17. Choosing `4. View Leaderboards` shows the ranked users by overall performance.
18. Choosing `5. Logout to Authentication Menu` returns you to the authentication menu without closing the app.
19. Choosing `6. Exit Application` closes the program completely.

## Leaderboards

The leaderboard ranks users using saved training records in `progress.txt`. It shows each user's total score, best score, session count, success rate, and average accuracy.

## File Handling

The program uses `progress.txt` to:

- save technique practiced per logged-in user through `start_training()` and `save_progress()` in `main.py`
- save repetitions, successful repetitions, accuracy percentage, score, and match outcome through manual summary entry or `calculate_session_metrics()` and `save_progress()` in `main.py`, using webcam validation data produced by `process_stream()` in `openpose.py` when webcam mode is enabled
- allow each user to review only their own training sessions through `load_progress()` and `view_progress()` in `main.py`
- provide leaderboard rankings across all users based on saved session results through `build_leaderboard_entries()` and `view_leaderboards()` in `main.py`

The program also uses `users.json` to store local account records for registration and login.

The system may also keep internal validation values for compatibility, but the user-facing workflow focuses on one final session summary block with the technique, session status, match result, matched repetitions, match coverage, and training score. The progress file is created, normalized, and safely read through `ensure_progress_file_format()` and `load_all_progress()` in `main.py`.

## Error Handling

The system prevents crashes by validating:

- menu input through `get_auth_choice()`, `get_menu_choice()`, and `ask_yes_no()` in `main.py`
- technique selection through `choose_technique()` in `main.py`
- automatic default webcam selection for webcam validation through `get_detected_cameras()`, `probe_cameras_with_venv()`, and the camera-setup path inside `start_training()` in `main.py`, with backend probing in `discover_working_cameras()` in `openpose.py`
- username and password entry during authentication through `prompt_username()`, `prompt_password()`, `validate_username()`, `register_user()`, and `login_user()` in `main.py`
- back/cancel commands during training prompts through `is_back_command()` and `is_exit_command()` in `main.py`, which are used by the input flow functions

It also handles missing or empty progress files safely through `ensure_progress_file_format()`, `load_all_progress()`, and `load_users()` in `main.py`. Webcam and runtime failures are handled through `is_missing_cv2_error()` and `run_webcam_validation_with_venv()` in `main.py`, plus `build_capture_error()`, `load_net()`, `process_stream()`, and `fail()` in `openpose.py`.

## Visual Feedback

- The terminal groups output into clear phases such as technique choice, validation setup, and session summary.
- Important statuses such as `Passed`, `Needs Improvement`, `Matched`, `Near Match`, and `Not Matched` are highlighted with their own colors when the terminal supports ANSI formatting.
- The OpenCV webcam window uses dashboard-style panels, confidence bars, and color-coded alert cards for correct form, adjustment prompts, dark camera input, and unclear pose detection.

## Course Deliverables

- `PROJECT_DOCUMENTATION.md` contains the required documentation sections for submission.
- `PRESENTATION_SLIDES.md` contains slide-by-slide presentation content for the final demonstration.
- `PRESENTATION_SLIDES.html` provides a presentation file that can be opened directly in a browser.
- `REQUIREMENTS_CHECKLIST.md` maps the course rubric to the current system implementation.

