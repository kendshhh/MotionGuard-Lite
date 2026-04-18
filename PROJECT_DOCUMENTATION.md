# MotionGuard Lite Project Documentation

## Project Title

MotionGuard Lite: A Python-Based Self-Defense Training Guide and Progress Tracker

## Group Members

Fill in your actual group members before submission:

1. ____________________
2. ____________________
3. ____________________
4. ____________________

## Brief Description of the System

MotionGuard Lite is a Python-based terminal system that helps users practice basic self-defense techniques such as Block, Punch, and Escape. The system provides a menu-driven terminal workflow, technique guidance, progress tracking, and optional OpenCV-based webcam validation with real-time dashboard feedback.

## Objectives of the Project

1. To build a practical Python-based terminal application that solves a real-world training and progress-tracking problem.
2. To demonstrate the correct and meaningful use of core Python programming concepts required in the course.
3. To provide users with guided technique practice, progress recording, and robust handling of invalid input and runtime errors.

## Features and Functionalities

1. Menu-driven terminal interface
2. Technique selection for Block, Punch, and Escape
3. Step-by-step technique guide
4. Session metrics and score tracking based on saved training results
5. Automatic score and accuracy computation for both webcam-validated sessions and manual-summary sessions
6. Progress history saved to a text file
7. Optional OpenCV webcam validation with technique-specific dashboard feedback
8. Manual summary entry when webcam validation is skipped
9. Fullscreen-capable OpenCV feedback window with live status cards and confidence indicators
10. Input validation, exception handling, and back/cancel navigation

## Programming Concepts Applied

### 1. Data Types

The project uses the following Python data types:

- Integers: menu choices, repetitions, successful repetitions, frame counts
- Floats: confidence values, accuracy percentages, inference time, technique match ratio
- Strings: titles, prompts, status labels, descriptions, feedback text, file contents
- Booleans: pass/fail status, webcam validation choice, stance-kept state, pose/technique match state

### 2. Operators and Control Structures

The system uses:

- Arithmetic operators such as `+`, `-`, `*`, `/` for score calculation, distances, and percentages
- Relational operators such as `<`, `<=`, `>=`, `==` for validation logic and technique classification
- Logical operators such as `and`, `or`, `not` for combining pose conditions and user-input checks
- Conditional statements using `if`, `elif`, and `else` throughout the program for menu flow, validation, scoring, and webcam feedback

### 3. Loops

The system uses:

- `while` loops for the main menu, repeated input validation, and the webcam processing loop
- `for` loops for displaying techniques, iterating over saved progress records, drawing pose pairs, and building feedback output

### 4. Functions

The project is organized into reusable functions such as:

- `show_menu()`
- `get_menu_choice()`
- `choose_technique()`
- `start_training()`
- `view_techniques()`
- `view_progress()`
- `calculate_session_metrics()`
- `analyze_stream_source()`
- `build_pose_feedback()`
- `draw_technique_info()`

These functions make the code modular, readable, and easier to maintain.

### 5. List Handling

The project uses lists for:

- storing the available techniques and their steps in `TECHNIQUES`
- storing progress records loaded from `progress.txt`
- storing live pose feedback lines shown in the OpenCV window
- storing pose pairs and technique names used in detection logic

### 6. File Handling

The project demonstrates file handling through:

- creating `progress.txt` if it does not exist
- reading saved records from `progress.txt`
- appending new training records to `progress.txt`
- creating, reading, and updating `users.json` for local account storage
- writing temporary JSON validation summaries when webcam validation falls back to the project virtual environment

### 7. Error Handling

The system uses error handling through:

- menu and numeric input validation
- range checking for menu options and camera-related choices
- `try` / `except` blocks for file access and webcam validation
- handling missing OpenCV or wrong Python interpreter cases
- handling missing model files and invalid webcam or camera-backend issues
- preventing crashes due to invalid input and missing files

## Explanation of File Handling Used

The main text file used by the system is `progress.txt`. It stores the user's timestamp, username, technique, repetitions, successful repetitions, accuracy, score, status, detected technique, recognition confidence, and match result. The program creates the file automatically when needed, reads existing data safely, migrates older progress-file formats when necessary, and appends new session results after each completed training run.

The system also uses `users.json` to store registered usernames together with password salts and password hashes for login authentication.

The webcam validation fallback also uses a temporary JSON file so that `main.py` can receive OpenCV validation results even if the active interpreter does not have the `cv2` package installed.

## Explanation of Error Handling Implemented

The system protects itself from invalid input and runtime problems in several ways:

1. Menu choices only accept valid numbered options.
2. Training prompts validate menu selections, username rules, and password confirmation, and they support back and cancel commands.
3. The program supports back and cancel flows instead of forcing invalid input.
4. File reading is wrapped in safe exception handling to avoid crashes when files are missing or unreadable.
5. Webcam validation catches OpenCV and interpreter errors and retries with the project virtual environment when needed.
6. Missing model files, camera access issues, and invalid pose states are handled with readable error messages instead of program failure.

## Instructions on How to Run the Program

### Recommended terminal run

```powershell
.\run_main.bat
```

### Direct Python run

```powershell
.\.venv\Scripts\python.exe main.py
```

### Install dependencies if needed

```powershell
.\install_requirements.bat
```

### Webcam validation only

```powershell
.\run_webcam.bat --target-technique Punch
```

## Submission Notes

Before final submission, update the following:

1. Replace the blank group-member fields with the actual names of your group members.
2. Review the README and this documentation file for any group-specific wording your instructor wants.
3. Use `PRESENTATION_SLIDES.html` as the presentation file, or convert it to the exact slide format your instructor requires.
4. Make sure the model file `models/graph_opt.pb` is included in the submission package.
