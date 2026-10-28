import time
import ctypes
from ctypes import wintypes
from PIL import Image
from mss import mss
import pygetwindow as gw

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# Constantes Win32 para captura e interacción en segundo plano (sin robar el primer plano)
PW_RENDERFULLCONTENT = 0x00000002
BI_RGB = 0
DIB_RGB_COLORS = 0
SW_SHOWNOACTIVATE = 4

WM_ACTIVATE = 0x0006
WA_ACTIVE = 1
WM_SETFOCUS = 0x0007
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_CHAR = 0x0102
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
MK_LBUTTON = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

VK_MAP = {
    "left": (0x25, True),
    "up": (0x26, True),
    "right": (0x27, True),
    "down": (0x28, True),
    "space": (0x20, False),
    "escape": (0x1B, False),
    "esc": (0x1B, False),
    "enter": (0x0D, False),
    "return": (0x0D, False),
    "backspace": (0x08, False),
    "delete": (0x2E, True),
    "tab": (0x09, False),
}


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 3),
    ]


class WindowHandler:
    def get_window(self, title="Rise of Kingdoms"):
        windows = gw.getWindowsWithTitle(title)
        if not windows:
            print(f"No window found with title: {title}")
            return None
        return windows[0]

    def _get_hwnd(self, title="Rise of Kingdoms"):
        win = self.get_window(title)
        if win is None:
            return None, None
        hwnd = getattr(win, "_hWnd", None)
        if hwnd and user32.IsIconic(hwnd):
            # Si fue minimizada a la barra de tareas, restaurarla sin activarla ni traerla al frente
            user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
            time.sleep(0.15)
        return hwnd, win

    def _keep_unity_focused_in_background(self, hwnd):
        """
        Envía WM_ACTIVATE(WA_ACTIVE) y WM_SETFOCUS directamente al WndProc del juego
        SIN llamar a SetForegroundWindow, para que Unity procese teclas y clics aunque
        el juego esté detrás de otras ventanas del usuario.
        """
        if not hwnd:
            return
        user32.SendMessageW(hwnd, WM_ACTIVATE, WA_ACTIVE, 0)
        user32.SendMessageW(hwnd, WM_SETFOCUS, 0, 0)

    def screenshot_window(self, title="Rise of Kingdoms"):
        """
        Captura la ventana del juego usando PrintWindow (PW_RENDERFULLCONTENT) para que
        funcione incluso cuando el usuario tiene otras ventanas abiertas encima del juego.
        Si PrintWindow falla, usa mss() como respaldo.
        """
        hwnd, win = self._get_hwnd(title)
        if not win or not hwnd:
            return None, None

        w = int(win.width)
        h = int(win.height)
        if w <= 0 or h <= 0:
            return None, None

        hwnd_dc = 0
        mem_dc = 0
        bitmap = 0
        old_bmp = 0
        try:
            hwnd_dc = user32.GetWindowDC(hwnd)
            mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
            bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
            old_bmp = gdi32.SelectObject(mem_dc, bitmap)

            result = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
            if result == 1:
                bmi = BITMAPINFO()
                bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                bmi.bmiHeader.biWidth = w
                bmi.bmiHeader.biHeight = -h  # Top-down DIB
                bmi.bmiHeader.biPlanes = 1
                bmi.bmiHeader.biBitCount = 32
                bmi.bmiHeader.biCompression = BI_RGB

                buf_len = w * h * 4
                buffer = ctypes.create_string_buffer(buf_len)
                lines = gdi32.GetDIBits(mem_dc, bitmap, 0, h, buffer, ctypes.byref(bmi), DIB_RGB_COLORS)
                if lines == h:
                    screenshot = Image.frombuffer("RGB", (w, h), buffer, "raw", "BGRX", 0, 1).copy()
                    # Verificar que no sea una captura completamente negra
                    extrema = screenshot.getextrema()
                    if any(ch_max > 0 for (_, ch_max) in extrema):
                        return screenshot, win
        except Exception:
            pass
        finally:
            if old_bmp and mem_dc:
                gdi32.SelectObject(mem_dc, old_bmp)
            if bitmap:
                gdi32.DeleteObject(bitmap)
            if mem_dc:
                gdi32.DeleteDC(mem_dc)
            if hwnd_dc and hwnd:
                user32.ReleaseDC(hwnd, hwnd_dc)

        # Respaldo con mss() si PrintWindow no estuvo disponible
        sct = mss()
        monitor = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
        try:
            img = sct.grab(monitor)
        finally:
            sct.close()
        screenshot = Image.frombytes("RGB", img.size, img.rgb, "raw")
        return screenshot, win

    def activate_window(self, title="Rise of Kingdoms"):
        """
        Prepara la ventana para recibir eventos en segundo plano SIN traerla al frente
        (no usa win.activate() ni SetForegroundWindow para no interrumpir al usuario).
        """
        try:
            hwnd, _ = self._get_hwnd(title)
            if hwnd:
                self._keep_unity_focused_in_background(hwnd)
        except Exception:
            pass

    def click_screen(self, screen_x, screen_y, title="Rise of Kingdoms", hold_time=0.055):
        """
        Realiza un clic confiable en Rise of Kingdoms (Unity) en (screen_x, screen_y):
        - Si el punto (screen_x, screen_y) pertenece a la ventana visible del juego, utiliza
          mouse_event (MOUSEEVENTF_LEFTDOWN / MOUSEEVENTF_LEFTUP) manteniendo el cursor el tiempo
          de fotograma necesario para el EventSystem de Unity y restaurando al instante la ventana
          activa previa y la posición del ratón del usuario.
        - Si otra ventana del usuario está cubriendo exactamente el punto (screen_x, screen_y),
          envía los mensajes Win32 directamente al HWND del juego sin hacer clic en la ventana
          superpuesta del usuario.
        """
        hwnd, win = self._get_hwnd(title)
        if not hwnd or not win:
            return False

        target_x = int(screen_x)
        target_y = int(screen_y)

        pt = POINT(target_x, target_y)
        user32.ScreenToClient(hwnd, ctypes.byref(pt))
        lparam = ((pt.y & 0xFFFF) << 16) | (pt.x & 0xFFFF)

        # Si el usuario está presionando el botón izquierdo físico, esperar un instante
        wait_t0 = time.time()
        while (user32.GetAsyncKeyState(0x01) & 0x8000) and (time.time() - wait_t0 < 0.25):
            time.sleep(0.02)

        prev_fg_hwnd = user32.GetForegroundWindow()
        old_cursor = POINT()
        user32.GetCursorPos(ctypes.byref(old_cursor))

        pt_screen = POINT(target_x, target_y)
        hwnd_at_point = user32.WindowFromPoint(pt_screen)
        point_is_unobstructed = (
            hwnd_at_point == hwnd
            or bool(user32.IsChild(hwnd, hwnd_at_point))
        )

        try:
            self._keep_unity_focused_in_background(hwnd)
            user32.SetCursorPos(target_x, target_y)

            if point_is_unobstructed:
                # Punto visible en pantalla: usar mouse_event nativo para 100% compatibilidad con Unity UI
                time.sleep(0.040)
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(max(0.050, hold_time))
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                time.sleep(0.045)
            else:
                # Punto cubierto por otra ventana: enviar mensajes directos al HWND sin clicar la ventana superior
                user32.SendMessageW(hwnd, WM_MOUSEMOVE, 0, lparam)
                time.sleep(0.040)
                user32.SendMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
                time.sleep(max(0.055, hold_time))
                user32.SendMessageW(hwnd, WM_LBUTTONUP, 0, lparam)
                time.sleep(0.050)
        finally:
            user32.SetCursorPos(old_cursor.x, old_cursor.y)
            if prev_fg_hwnd and prev_fg_hwnd != hwnd:
                try:
                    if user32.GetForegroundWindow() != prev_fg_hwnd:
                        user32.SetForegroundWindow(prev_fg_hwnd)
                except Exception:
                    pass

        return True

    def _resolve_vk(self, key_name):
        k = key_name.lower()
        if k in VK_MAP:
            return VK_MAP[k]
        if len(k) == 1:
            vk = user32.VkKeyScanW(ord(k)) & 0xFF
            return vk, False
        return 0, False

    def _make_key_lparam(self, vk, is_extended=False, is_keyup=False, is_repeat=False):
        scan = user32.MapVirtualKeyW(vk, 0) & 0xFF
        lparam = 1 | (scan << 16)
        if is_extended:
            lparam |= 1 << 24
        if is_repeat or is_keyup:
            lparam |= 1 << 30
        if is_keyup:
            lparam |= 1 << 31
        return lparam

    def hold_key(self, key_name, duration=0.35, title="Rise of Kingdoms"):
        """
        Mantiene presionada una tecla (ej. 'left', 'right', 'up', 'down', 'space') directamente
        en el HWND del juego durante 'duration' segundos sin robar el foco del teclado del usuario.
        """
        hwnd, _ = self._get_hwnd(title)
        if not hwnd:
            return False

        vk, is_ext = self._resolve_vk(key_name)
        if not vk:
            return False

        lparam_down = self._make_key_lparam(vk, is_ext, is_keyup=False, is_repeat=False)
        lparam_repeat = self._make_key_lparam(vk, is_ext, is_keyup=False, is_repeat=True)
        lparam_up = self._make_key_lparam(vk, is_ext, is_keyup=True)

        self._keep_unity_focused_in_background(hwnd)
        user32.PostMessageW(hwnd, WM_KEYDOWN, vk, lparam_down)

        end_time = time.time() + duration
        while time.time() < end_time:
            time.sleep(min(0.025, max(0.0, end_time - time.time())))
            self._keep_unity_focused_in_background(hwnd)
            user32.PostMessageW(hwnd, WM_KEYDOWN, vk, lparam_repeat)

        user32.PostMessageW(hwnd, WM_KEYUP, vk, lparam_up)
        return True

    def press_key(self, key_name, presses=1, interval=0.02, title="Rise of Kingdoms"):
        """
        Pulsa una tecla una o varias veces en segundo plano en el HWND del juego.
        """
        hwnd, _ = self._get_hwnd(title)
        if not hwnd:
            return False

        vk, is_ext = self._resolve_vk(key_name)
        if not vk:
            return False

        lparam_down = self._make_key_lparam(vk, is_ext, is_keyup=False, is_repeat=False)
        lparam_up = self._make_key_lparam(vk, is_ext, is_keyup=True)

        for _ in range(presses):
            self._keep_unity_focused_in_background(hwnd)
            user32.PostMessageW(hwnd, WM_KEYDOWN, vk, lparam_down)
            if len(key_name) == 1:
                user32.PostMessageW(hwnd, WM_CHAR, ord(key_name), lparam_down)
            time.sleep(0.015)
            user32.PostMessageW(hwnd, WM_KEYUP, vk, lparam_up)
            if interval > 0:
                time.sleep(interval)

        return True

    def write_text(self, text, interval=0.03, title="Rise of Kingdoms"):
        """
        Escribe una cadena de texto (ej. '10000') directamente en el HWND del juego en segundo plano.
        """
        hwnd, _ = self._get_hwnd(title)
        if not hwnd:
            return False

        for ch in str(text):
            vk, is_ext = self._resolve_vk(ch)
            lparam_down = self._make_key_lparam(vk, is_ext, is_keyup=False) if vk else 1
            lparam_up = self._make_key_lparam(vk, is_ext, is_keyup=True) if vk else (1 | (3 << 30))

            self._keep_unity_focused_in_background(hwnd)
            if vk:
                user32.PostMessageW(hwnd, WM_KEYDOWN, vk, lparam_down)
            user32.PostMessageW(hwnd, WM_CHAR, ord(ch), lparam_down)
            time.sleep(0.01)
            if vk:
                user32.PostMessageW(hwnd, WM_KEYUP, vk, lparam_up)
            if interval > 0:
                time.sleep(interval)

        return True
