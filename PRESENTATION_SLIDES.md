# MotionGuard Lite Presentation Slides

## Slide 1: Project Title

MotionGuard Lite: A Python-Based Self-Defense Training Guide and Progress Tracker

Group Members:

1. ____________________
2. ____________________
3. ____________________
4. ____________________

## Slide 2: Project Purpose

- Provide a Python-based training tool for guided self-defense practice
- Combine terminal interaction, file handling, and optional OpenCV webcam validation
- Solve a practical problem by helping users practice techniques and track progress

## Slide 3: System Objectives

- Build a functional menu-driven Python program
- Demonstrate all required programming concepts in one complete system
- Handle invalid input and runtime errors safely
- Save and review training performance over time

## Slide 4: Main Features

- Guided training for Block, Punch, and Escape
- Technique guide and progress review
- Real-time webcam validation with live feedback
- Manual summary entry when webcam validation is skipped
- Fullscreen OpenCV display controls
- Automatic score calculation and progress saving

## Slide 5: Programming Concepts Applied

- Data types: int, float, str, bool
- Operators and control structures
- Loops: `for` and `while`
- Functions
- List handling
- File handling
- Error handling

## Slide 6: File Handling Used

- Creates `progress.txt` when needed
- Reads saved training records
- Writes and updates training results
- Creates and updates `users.json` for account storage
- Uses temporary JSON files for webcam-validation fallback communication

## Slide 7: Error Handling Implemented

- Menu input validation
- Authentication validation and back/cancel support
- Exception handling with `try` and `except`
- Missing-file protection
- Webcam and camera-backend error handling
- OpenCV fallback through the project virtual environment

## Slide 8: Demo Flow

1. Open the terminal app
2. Choose `1. Start Training`
3. Select a technique
4. Choose whether to use webcam validation
5. If Yes, perform the move and review live feedback
6. If No, enter repetitions and successful repetitions manually
7. Review the session summary and saved progress

## Slide 9: Invalid Input / Error Demo

- Show invalid menu input handling
- Show invalid login or registration input handling
- Show back/cancel navigation
- Show webcam fallback or pose-not-clear handling

## Slide 10: Conclusion

- The project is functional, practical, and well organized
- It demonstrates all required Python programming concepts
- It shows how Python can solve a real-world training and tracking problem
