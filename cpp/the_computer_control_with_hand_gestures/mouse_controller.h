
#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <algorithm>
#include <cmath> // Needed for sqrt()

class MouseController {
private:
    static inline float currentCursorX = 0.0f;
    static inline float currentCursorY = 0.0f;

    // Trackpad Memory
    static inline float lastRawX = -1.0f;
    static inline float lastRawY = -1.0f;
    static inline float alpha = 0.40f;

    // Pointer Acceleration Tuning
    static inline float baseSensitivity = 1.2f;   // Speed when moving slowly
    static inline float accelMultiplier = 40.0f;  // How aggressively it speeds up
    static inline float maxSensitivity = 6.0f;    // Speed limit for fast flicks
    static inline float deadzone = 0.001f;        // Ignores tiny natural hand shakes

    static inline bool isDragging = false;
    static inline int screenW = GetSystemMetrics(SM_CXSCREEN);
    static inline int screenH = GetSystemMetrics(SM_CYSCREEN);

public:
    static void moveCursor(float rawX, float rawY) {
        if (lastRawX < 0.0f || lastRawY < 0.0f) {
            lastRawX = rawX;
            lastRawY = rawY;
            POINT p;
            if (GetCursorPos(&p)) {
                currentCursorX = (float)p.x;
                currentCursorY = (float)p.y;
            }
            return;
        }

        // 1. Get raw distance moved
        float deltaX = rawX - lastRawX;
        float deltaY = rawY - lastRawY;

        lastRawX = rawX;
        lastRawY = rawY;

        // 2. Calculate Hand Speed
        float speed = std::sqrt(deltaX * deltaX + deltaY * deltaY);

        // 3. Deadzone: If moving extremely slowly, ignore it (stops jitter)
        if (speed < deadzone) {
            return;
        }

        // 4. DYNAMIC ACCELERATION: Slow hand = Low Multiplier. Fast hand = High Multiplier.
        float dynamicSensitivity = baseSensitivity + (speed * accelMultiplier);
        dynamicSensitivity = std::clamp(dynamicSensitivity, baseSensitivity, maxSensitivity);

        float moveX = deltaX * screenW * dynamicSensitivity;
        float moveY = deltaY * screenH * dynamicSensitivity;

        // 5. Apply limits and smoothing
        float targetX = std::clamp(currentCursorX + moveX, 0.0f, (float)screenW);
        float targetY = std::clamp(currentCursorY + moveY, 0.0f, (float)screenH);

        currentCursorX = (targetX * alpha) + (currentCursorX * (1.0f - alpha));
        currentCursorY = (targetY * alpha) + (currentCursorY * (1.0f - alpha));

        SetCursorPos((int)currentCursorX, (int)currentCursorY);
    }

    static void resetTracking() {
        lastRawX = -1.0f;
        lastRawY = -1.0f;
    }

    // --- LEFT & RIGHT CLICKS ---
    static void leftClick() {
        INPUT inputs[2] = { 0 };
        inputs[0].type = INPUT_MOUSE; inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        inputs[1].type = INPUT_MOUSE; inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(2, inputs, sizeof(INPUT));
    }

    static void rightClick() {
        INPUT inputs[2] = { 0 };
        inputs[0].type = INPUT_MOUSE; inputs[0].mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
        inputs[1].type = INPUT_MOUSE; inputs[1].mi.dwFlags = MOUSEEVENTF_RIGHTUP;
        SendInput(2, inputs, sizeof(INPUT));
    }

    static void doubleClick() { leftClick(); Sleep(50); leftClick(); }
    static void scroll(int amount) {
        INPUT input = { 0 };
        input.type = INPUT_MOUSE; input.mi.dwFlags = MOUSEEVENTF_WHEEL; input.mi.mouseData = amount;
        SendInput(1, &input, sizeof(INPUT));
    }
    static void holdLeftDown() {
        if (!isDragging) {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE; input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
            SendInput(1, &input, sizeof(INPUT));
            isDragging = true;
        }
    }
    static void releaseLeft() {
        if (isDragging) {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE; input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
            isDragging = false;
        }
    }
    static void pressKey(WORD virtualKey) {
        INPUT inputs[2] = { 0 };
        inputs[0].type = INPUT_KEYBOARD; inputs[0].ki.wVk = virtualKey;
        inputs[1].type = INPUT_KEYBOARD; inputs[1].ki.wVk = virtualKey; inputs[1].ki.dwFlags = KEYEVENTF_KEYUP;
        SendInput(2, inputs, sizeof(INPUT));
    }
    static void volumeUp() { pressKey(VK_VOLUME_UP); }
    static void volumeDown() { pressKey(VK_VOLUME_DOWN); }
    static void muteVolume() { pressKey(VK_VOLUME_MUTE); }
    static void showDesktop() {
        INPUT inputs[4] = { 0 };
        inputs[0].type = INPUT_KEYBOARD; inputs[0].ki.wVk = VK_LWIN;
        inputs[1].type = INPUT_KEYBOARD; inputs[1].ki.wVk = 'D';
        inputs[2].type = INPUT_KEYBOARD; inputs[2].ki.wVk = 'D'; inputs[2].ki.dwFlags = KEYEVENTF_KEYUP;
        inputs[3].type = INPUT_KEYBOARD; inputs[3].ki.wVk = VK_LWIN; inputs[3].ki.dwFlags = KEYEVENTF_KEYUP;
        SendInput(4, inputs, sizeof(INPUT));
    }
};
