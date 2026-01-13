#pragma once
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// Simple CPU-side plotter for burning graphs into pixel buffers
class SimplePlotter {
   public:
    static void PlotLine(std::vector<unsigned char>& image,
                         int imgWidth,
                         int imgHeight,
                         const std::vector<float>& data,
                         float yMin,
                         float yMax,
                         int x,
                         int y,
                         int w,
                         int h,
                         const std::string& title,
                         unsigned char r,
                         unsigned char g,
                         unsigned char b) {
        // Style Constants
        const int paddingLeft = 35;
        const int paddingRight = 10;
        const int paddingTop = 20;
        const int paddingBottom = 10;

        int graphX = x + paddingLeft;
        int graphY = y + paddingTop;
        int graphW = w - paddingLeft - paddingRight;
        int graphH = h - paddingTop - paddingBottom;

        // Draw background box (Semi-transparent dark)
        FillBox(image, imgWidth, imgHeight, x, y, w, h, 20, 20, 30, 230);

        // Title
        int titleLen = title.length() * 6;  // 5px char + 1px spacing
        int titleX = x + (w - titleLen) / 2;
        DrawString(image, imgWidth, imgHeight, title, titleX, y + 5, 255, 255, 255);

        // Grid lines
        int numHGrid = 5;
        for (int i = 0; i <= numHGrid; ++i) {
            int gy = graphY + (graphH * i) / numHGrid;
            DrawLine(image, imgWidth, imgHeight, graphX, gy, graphX + graphW, gy, 60, 60, 60);
        }
        int numVGrid = 5;
        for (int i = 0; i <= numVGrid; ++i) {
            int gx = graphX + (graphW * i) / numVGrid;
            DrawLine(image, imgWidth, imgHeight, gx, graphY, gx, graphY + graphH, 60, 60, 60);
        }

        // Labels (Min/Max)
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << yMax;
        DrawString(image, imgWidth, imgHeight, ss.str(), x + 2, graphY - 3, 200, 200, 200);

        ss.str("");
        ss << std::fixed << std::setprecision(1) << yMin;
        DrawString(image, imgWidth, imgHeight, ss.str(), x + 2, graphY + graphH - 4, 200, 200, 200);

        // Plot data
        if (data.size() >= 2) {
            float xStep = (float) graphW / (data.size() - 1);
            float yRange = yMax - yMin;
            if (std::abs(yRange) < 1e-5)
                yRange = 1.0f;

            for (size_t i = 0; i < data.size() - 1; ++i) {
                float val1 = data[i];
                float val2 = data[i + 1];

                // Normalize and clamp
                float ny1 = (val1 - yMin) / yRange;
                float ny2 = (val2 - yMin) / yRange;
                ny1 = std::max(0.0f, std::min(1.0f, ny1));
                ny2 = std::max(0.0f, std::min(1.0f, ny2));

                int px1 = graphX + (int) (i * xStep);
                int py1 = graphY + graphH - (int) (ny1 * graphH);
                int px2 = graphX + (int) ((i + 1) * xStep);
                int py2 = graphY + graphH - (int) (ny2 * graphH);

                DrawLine(image, imgWidth, imgHeight, px1, py1, px2, py2, r, g, b);
            }
        }

        // Graph border
        DrawRect(image, imgWidth, imgHeight, graphX, graphY, graphW, graphH, 120, 120, 120);

        // Outer border
        DrawRect(image, imgWidth, imgHeight, x, y, w, h, 100, 100, 100);
    }

   private:
    static void DrawLine(std::vector<unsigned char>& img,
                         int w,
                         int h,
                         int x0,
                         int y0,
                         int x1,
                         int y1,
                         unsigned char r,
                         unsigned char g,
                         unsigned char b) {
        int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy, e2;

        while (true) {
            if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h) {
                int idx = (y0 * w + x0) * 4;
                img[idx] = r;
                img[idx + 1] = g;
                img[idx + 2] = b;
                img[idx + 3] = 255;
            }
            if (x0 == x1 && y0 == y1)
                break;
            e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx) {
                err += dx;
                y0 += sy;
            }
        }
    }

    static void FillBox(std::vector<unsigned char>& img,
                        int imgW,
                        int imgH,
                        int rx,
                        int ry,
                        int rw,
                        int rh,
                        unsigned char r,
                        unsigned char g,
                        unsigned char b,
                        unsigned char a) {
        for (int y = ry; y < ry + rh; ++y) {
            for (int x = rx; x < rx + rw; ++x) {
                if (x >= 0 && x < imgW && y >= 0 && y < imgH) {
                    int idx = (y * imgW + x) * 4;
                    // Blend
                    unsigned char oldR = img[idx];
                    unsigned char oldG = img[idx + 1];
                    unsigned char oldB = img[idx + 2];

                    float alpha = a / 255.0f;
                    img[idx] = (unsigned char) (r * alpha + oldR * (1.0f - alpha));
                    img[idx + 1] = (unsigned char) (g * alpha + oldG * (1.0f - alpha));
                    img[idx + 2] = (unsigned char) (b * alpha + oldB * (1.0f - alpha));
                    img[idx + 3] = 255;
                }
            }
        }
    }

    static void DrawRect(std::vector<unsigned char>& img,
                         int w,
                         int h,
                         int rx,
                         int ry,
                         int rw,
                         int rh,
                         unsigned char r,
                         unsigned char g,
                         unsigned char b) {
        // Top/Bottom
        for (int cx = rx; cx < rx + rw; ++cx) {
            DrawLine(img, w, h, cx, ry, cx, ry, r, g, b);
            DrawLine(img, w, h, cx, ry + rh - 1, cx, ry + rh - 1, r, g, b);
        }
        // Left/Right
        for (int cy = ry; cy < ry + rh; ++cy) {
            DrawLine(img, w, h, rx, cy, rx, cy, r, g, b);
            DrawLine(img, w, h, rx + rw - 1, cy, rx + rw - 1, cy, r, g, b);
        }
    }

    static void DrawString(std::vector<unsigned char>& img,
                           int w,
                           int h,
                           const std::string& text,
                           int x,
                           int y,
                           unsigned char r,
                           unsigned char g,
                           unsigned char b) {
        int cursorX = x;
        for (char c : text) {
            DrawChar(img, w, h, c, cursorX, y, r, g, b);
            cursorX += 6;  // 5 width + 1 spacing
        }
    }

    static void DrawChar(std::vector<unsigned char>& img,
                         int w,
                         int h,
                         char c,
                         int x,
                         int y,
                         unsigned char r,
                         unsigned char g,
                         unsigned char b) {
        static const unsigned char font[43][5] = {
            {0x3E, 0x51, 0x49, 0x45, 0x3E},  // 0
            {0x00, 0x42, 0x7F, 0x40, 0x00},  // 1
            {0x42, 0x61, 0x51, 0x49, 0x46},  // 2
            {0x21, 0x41, 0x45, 0x4B, 0x31},  // 3
            {0x18, 0x14, 0x12, 0x7F, 0x10},  // 4
            {0x27, 0x45, 0x45, 0x45, 0x39},  // 5
            {0x3C, 0x4A, 0x49, 0x49, 0x30},  // 6
            {0x01, 0x71, 0x09, 0x05, 0x03},  // 7
            {0x36, 0x49, 0x49, 0x49, 0x36},  // 8
            {0x06, 0x49, 0x49, 0x29, 0x1E},  // 9
            {0x00, 0x00, 0x60, 0x60, 0x00},  // . (10)
            {0x08, 0x08, 0x08, 0x08, 0x08},  // - (11)
            {0x00, 0x00, 0x00, 0x00, 0x00},  // Space (12)
            {0x7E, 0x11, 0x11, 0x11, 0x7E},  // A
            {0x7F, 0x49, 0x49, 0x49, 0x36},  // B
            {0x3E, 0x41, 0x41, 0x41, 0x22},  // C
            {0x7F, 0x41, 0x41, 0x22, 0x1C},  // D
            {0x7F, 0x49, 0x49, 0x49, 0x41},  // E
            {0x7F, 0x09, 0x09, 0x09, 0x01},  // F
            {0x3E, 0x41, 0x49, 0x49, 0x7A},  // G
            {0x7F, 0x08, 0x08, 0x08, 0x7F},  // H
            {0x00, 0x41, 0x7F, 0x41, 0x00},  // I
            {0x20, 0x40, 0x41, 0x3F, 0x01},  // J
            {0x7F, 0x08, 0x14, 0x22, 0x41},  // K
            {0x7F, 0x40, 0x40, 0x40, 0x40},  // L
            {0x7F, 0x02, 0x0C, 0x02, 0x7F},  // M
            {0x7F, 0x04, 0x08, 0x10, 0x7F},  // N
            {0x3E, 0x41, 0x41, 0x41, 0x3E},  // O
            {0x7F, 0x09, 0x09, 0x09, 0x06},  // P
            {0x3E, 0x41, 0x51, 0x21, 0x5E},  // Q
            {0x7F, 0x09, 0x19, 0x29, 0x46},  // R
            {0x46, 0x49, 0x49, 0x49, 0x31},  // S
            {0x01, 0x01, 0x7F, 0x01, 0x01},  // T
            {0x3F, 0x40, 0x40, 0x40, 0x3F},  // U
            {0x1F, 0x20, 0x40, 0x20, 0x1F},  // V
            {0x3F, 0x40, 0x38, 0x40, 0x3F},  // W
            {0x63, 0x14, 0x08, 0x14, 0x63},  // X
            {0x07, 0x08, 0x70, 0x08, 0x07},  // Y
            {0x61, 0x51, 0x49, 0x45, 0x43}   // Z
        };

        int index = 12;  // default space
        if (c >= '0' && c <= '9')
            index = c - '0';
        else if (c == '.')
            index = 10;
        else if (c == '-')
            index = 11;
        else if (c == ' ')
            index = 12;
        else if (c >= 'A' && c <= 'Z')
            index = 13 + (c - 'A');
        else if (c >= 'a' && c <= 'z')
            index = 13 + (c - 'a');

        for (int col = 0; col < 5; ++col) {
            unsigned char colBits = font[index][col];
            for (int bit = 0; bit < 7; ++bit) {
                if ((colBits >> bit) & 1) {
                    DrawLine(img, w, h, x + col, y + bit, x + col, y + bit, r, g, b);
                }
            }
        }
    }
};
