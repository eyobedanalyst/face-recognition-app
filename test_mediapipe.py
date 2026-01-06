import mediapipe as mp

print("MediaPipe version:", mp.__version__)
print("MediaPipe file location:", mp.__file__)
print("Has solutions?", hasattr(mp, 'solutions'))

if hasattr(mp, 'solutions'):
    print("✓ MediaPipe is working correctly!")
    print("Available solutions:", dir(mp.solutions))
else:
    print("✗ MediaPipe installation is broken")