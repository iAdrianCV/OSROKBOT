import os
import time
import cv2
import numpy as np
import pyautogui

from Actions.action import Action
from Actions.screenshot_action import ScreenshotAction
from Actions.extract_text_action import ExtractTextAction
from image_finder import ImageFinder
from window_handler import WindowHandler
from global_vars import GlobalVars

try:
    from termcolor import colored
except ImportError:
    def colored(text, color):
        return text


# Lista de comandantes en retorno (flecha naranja) que ya tienen su carga llena (botón March gris).
# Cada entrada es un dict: {"face": patch_bgr_22x22, "last_seen": timestamp, "created": timestamp}
_FULL_RETURNING_COMMANDERS = []

# Máximo de marchas detectado en el contador superior (ej. 5 en "5/5" o "4/5")
_KNOWN_MAX_MARCHES = 5

# Última coordenada de pantalla (screen_x, screen_y) donde se encontró y abrió el módulo de gemas actual
_LAST_GEM_CLICK_POS = None


def _resolve_media_path(relative_path):
    if os.path.exists(relative_path):
        return relative_path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate = os.path.join(repo_root, relative_path)
    if os.path.exists(candidate):
        return candidate
    return relative_path


def _extract_commander_face_patch(screenshot_cv, cmd_cx, cmd_cy, base_scale):
    """
    Extrae el rostro central del comandante (aprox 22x22 px escalado), evitando el texto F1..F5
    a la izquierda y evitando el ícono de estado abajo a la derecha.
    """
    h, w = screenshot_cv.shape[:2]
    rx_left = max(7, int(9 * base_scale[0]))
    rx_right = max(10, int(13 * base_scale[0]))
    ry_top = max(9, int(11 * base_scale[1]))
    ry_bot = max(9, int(11 * base_scale[1]))

    x1 = max(0, int(cmd_cx - rx_left))
    x2 = min(w, int(cmd_cx + rx_right))
    y1 = max(0, int(cmd_cy - ry_top))
    y2 = min(h, int(cmd_cy + ry_bot))
    return screenshot_cv[y1:y2, x1:x2].copy()


def _is_commander_marked_full(screenshot_cv, cmd_cx, cmd_cy, base_scale, refresh_seen=True):
    """
    Verifica con tolerancia al desplazamiento (±12 px) si el comandante en (cmd_cx, cmd_cy)
    fue marcado con carga llena (botón March gris).
    """
    if not _FULL_RETURNING_COMMANDERS:
        return False

    h, w = screenshot_cv.shape[:2]
    pad_x = max(18, int(23 * base_scale[0]))
    pad_y = max(18, int(23 * base_scale[1]))

    sx1 = max(0, int(cmd_cx - pad_x))
    sx2 = min(w, int(cmd_cx + pad_x))
    sy1 = max(0, int(cmd_cy - pad_y))
    sy2 = min(h, int(cmd_cy + pad_y))
    search_roi = screenshot_cv[sy1:sy2, sx1:sx2]

    if search_roi is None or search_roi.size == 0:
        return False

    now = time.time()
    for entry in _FULL_RETURNING_COMMANDERS:
        face_tpl = entry["face"]
        if face_tpl is None or face_tpl.size == 0:
            continue
        if face_tpl.shape[0] >= search_roi.shape[0] or face_tpl.shape[1] >= search_roi.shape[1]:
            continue

        res = cv2.matchTemplate(search_roi, face_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if float(max_val) >= 0.74:
            if refresh_seen:
                entry["last_seen"] = now
            return True

    return False


def _mark_commander_as_full(screenshot_cv, cmd_cx, cmd_cy, base_scale):
    """
    Marca a un comandante en retorno como lleno (Load máximo / botón March gris)
    para dejarlo regresar al castillo sin volver a seleccionarlo.
    """
    if _is_commander_marked_full(screenshot_cv, cmd_cx, cmd_cy, base_scale, refresh_seen=True):
        return

    face_patch = _extract_commander_face_patch(screenshot_cv, cmd_cx, cmd_cy, base_scale)
    if face_patch is not None and face_patch.size > 0:
        now = time.time()
        _FULL_RETURNING_COMMANDERS.append(
            {
                "face": face_patch,
                "last_seen": now,
                "created": now,
            }
        )
        print(
            colored(
                f"[GemMarch] Comandante en ({cmd_cx}, {cmd_cy}) marcado con carga llena (dejándolo regresar al castillo).",
                "yellow",
            )
        )


def _prune_full_commanders(sidebar_visible, any_returning_present):
    """
    Limpia de la lista de comandantes llenos a aquellos que ya llegaron al castillo
    (es decir, ya no aparecen con ícono naranja de retorno durante > 7 segundos mientras la barra es visible)
    o cuyo tiempo máximo absoluto (600s) expiró.
    """
    global _FULL_RETURNING_COMMANDERS
    if not _FULL_RETURNING_COMMANDERS:
        return

    now = time.time()
    kept = []
    for entry in _FULL_RETURNING_COMMANDERS:
        age = now - entry["created"]
        unseen_time = now - entry["last_seen"]
        if age > 600:
            continue
        # Si la barra lateral de marchas está visible y este comandante ya no aparece regresando por > 7s,
        # significa que ya llegó al castillo.
        if sidebar_visible and unseen_time > 7.5:
            print(colored("[GemMarch] Una tropa llena ya llegó al castillo (liberada de la lista de espera).", "cyan"))
            continue
        if not any_returning_present and sidebar_visible and unseen_time > 4.5:
            print(colored("[GemMarch] Tropa en retorno llegó al castillo.", "cyan"))
            continue
        kept.append(entry)

    _FULL_RETURNING_COMMANDERS = kept


def _classify_badge_patch(bgr_patch):
    """
    Clasifica un parche de ícono de estado (16x16..18x18) según sus colores RGB:
    - 'green': pasos verdes (en camino) o pico verde (recolectando)
    - 'orange': flecha naranja (regresando a la ciudad)
    - 'none': no es un ícono válido
    """
    if bgr_patch is None or bgr_patch.size == 0:
        return "none", 0.0, 0.0, 0.0

    b = bgr_patch[:, :, 0].astype(np.float32)
    g = bgr_patch[:, :, 1].astype(np.float32)
    r = bgr_patch[:, :, 2].astype(np.float32)

    # Símbolo blanco central (huellas blancas, pico blanco o flecha curva blanca)
    white_mask = (r > 190) & (g > 190) & (b > 190)
    white_ratio = float(np.mean(white_mask))

    # Círculo verde (pasos verdes o pico verde)
    green_mask = (g > r + 28) & (g > b + 22) & (g > 105)
    green_ratio = float(np.mean(green_mask))

    # Círculo naranja (flecha de regreso)
    orange_mask = (r > g + 28) & (g > b + 18) & (r > 120) & (~white_mask)
    orange_ratio = float(np.mean(orange_mask))

    if green_ratio >= 0.16 and white_ratio >= 0.04:
        return "green", green_ratio, orange_ratio, white_ratio

    if orange_ratio >= 0.20 and green_ratio <= 0.05 and white_ratio >= 0.05:
        return "orange", green_ratio, orange_ratio, white_ratio

    return "none", green_ratio, orange_ratio, white_ratio


def _analyze_sidebar_commanders(screenshot_cv, base_scale):
    """
    Escanea la columna derecha de comandantes (x: 94.5%..99.3%, y: 19.5%..72.0%)
    e identifica el estado exacto de cada tropa por su ícono:
    - 'marching': pasos verdes (en camino a las gemas -> OCUPADA)
    - 'gathering': pico verde (recolectando gemas -> OCUPADA)
    - 'returning_free': flecha naranja con espacio de carga (LIBRE para redirigir)
    - 'returning_full': flecha naranja pero ya sin capacidad de carga (botón March gris -> dejar regresar al castillo)

    Retorna un dict con las listas clasificadas y el total de marchas visibles en la barra.
    """
    h_full, w_full = screenshot_cv.shape[:2]
    x_min = int(w_full * 0.900)
    x_max = int(w_full * 0.995)
    y_min = int(h_full * 0.140)
    y_max = int(h_full * 0.900)
    roi = screenshot_cv[y_min:y_max, x_min:x_max]

    badge_templates = [
        ("returning", "Media/returning_icon.png", 0.72),
        ("returning", "Media/isreturning.png", 0.72),
        ("marching", "Media/marching_icon.png", 0.70),
        ("gathering", "Media/gathering_icon.png", 0.70),
    ]

    raw_detections = []

    for kind, rel_path, thresh in badge_templates:
        full_path = _resolve_media_path(rel_path)
        tpl = cv2.imread(full_path)
        if tpl is None:
            continue

        for mult in [1.0, 0.92, 1.08, 1.18, 1.28]:
            fx = base_scale[0] * mult
            fy = base_scale[1] * mult
            nw = int(tpl.shape[1] * fx)
            nh = int(tpl.shape[0] * fy)
            if nw < 10 or nh < 10 or nw >= roi.shape[1] or nh >= roi.shape[0]:
                continue

            resized = cv2.resize(tpl, (nw, nh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= thresh)

            for pt_y, pt_x in zip(*locs):
                sc = float(res[pt_y, pt_x])
                patch = roi[pt_y : pt_y + nh, pt_x : pt_x + nw]
                color_class, g_ratio, o_ratio, w_ratio = _classify_badge_patch(patch)

                # Validación estricta de color para no confundir nunca íconos verdes con el naranja de retorno
                if kind == "returning" and color_class != "orange":
                    continue
                if kind in ("marching", "gathering") and color_class != "green":
                    continue

                badge_cx = x_min + pt_x + nw // 2
                badge_cy = y_min + pt_y + nh // 2
                raw_detections.append((badge_cx, badge_cy, sc, kind, rel_path))

    # Agrupar detecciones por ranura vertical de comandante (separación ~53 px entre comandantes)
    raw_detections.sort(key=lambda item: item[2], reverse=True)
    slot_min_dist = max(28, int(38 * base_scale[1]))
    slots = []

    for badge_cx, badge_cy, sc, kind, rel_path in raw_detections:
        existing_slot = None
        for s in slots:
            if abs(badge_cy - s["badge_cy"]) < slot_min_dist:
                existing_slot = s
                break

        if existing_slot is None:
            slots.append(
                {
                    "badge_cx": badge_cx,
                    "badge_cy": badge_cy,
                    "score": sc,
                    "kind": kind,
                    "rel_path": rel_path,
                }
            )
        else:
            # Si en la misma ranura hubo coincidencia verde con buen score, priorizar verde por seguridad
            if existing_slot["kind"] == "returning" and kind in ("marching", "gathering") and sc >= existing_slot["score"] - 0.04:
                existing_slot["kind"] = kind
                existing_slot["score"] = sc
                existing_slot["rel_path"] = rel_path

    # Ordenar ranuras de arriba hacia abajo (F5 -> F1)
    slots.sort(key=lambda s: s["badge_cy"])

    marching_list = []
    gathering_list = []
    returning_free_list = []
    returning_full_list = []

    offset_x = int(17 * base_scale[0])
    offset_y = int(15 * base_scale[1])

    for s in slots:
        cmd_cx = s["badge_cx"] - offset_x
        cmd_cy = s["badge_cy"] - offset_y
        kind = s["kind"]

        if kind == "marching":
            marching_list.append((cmd_cx, cmd_cy, s["score"], s["rel_path"]))
        elif kind == "gathering":
            gathering_list.append((cmd_cx, cmd_cy, s["score"], s["rel_path"]))
        elif kind == "returning":
            if _is_commander_marked_full(screenshot_cv, cmd_cx, cmd_cy, base_scale, refresh_seen=True):
                returning_full_list.append((cmd_cx, cmd_cy, s["score"], s["rel_path"]))
            else:
                returning_free_list.append((cmd_cx, cmd_cy, s["score"], s["rel_path"]))

    total_commanders = len(slots)
    any_returning = (len(returning_free_list) + len(returning_full_list)) > 0
    _prune_full_commanders(sidebar_visible=(total_commanders > 0), any_returning_present=any_returning)

    global _KNOWN_MAX_MARCHES
    if total_commanders > _KNOWN_MAX_MARCHES and total_commanders <= 7:
        _KNOWN_MAX_MARCHES = total_commanders

    return {
        "marching": marching_list,
        "gathering": gathering_list,
        "returning_free": returning_free_list,
        "returning_full": returning_full_list,
        "total_commanders": total_commanders,
    }


def _read_march_counter(screenshot_cv, total_sidebar_commanders=0):
    """
    Lee el contador de marchas en la cabeza de la lista derecha (ej. 5/5 o 4/5).
    Combina OCR sobre la región exacta (x: 95.5%..98.2%, y: 17.8%..20.0%) con el conteo
    de iconos de comandantes en la barra lateral para máxima fiabilidad.
    Retorna: (has_free_slot: bool, cur_marches: int, max_marches: int)
    """
    global _KNOWN_MAX_MARCHES

    # Si no hay ningún comandante en la barra derecha, verificar si todas las marchas están en el castillo (0/5)
    h, w = screenshot_cv.shape[:2]
    crop_x1 = int(w * 0.955)
    crop_x2 = int(w * 0.982)
    crop_y1 = int(h * 0.178)
    crop_y2 = int(h * 0.200)

    header_patch = screenshot_cv[crop_y1:crop_y2, crop_x1:crop_x2]
    if header_patch is None or header_patch.size == 0:
        has_free = total_sidebar_commanders < _KNOWN_MAX_MARCHES
        return has_free, total_sidebar_commanders, _KNOWN_MAX_MARCHES

    # Verificar si hay píxeles blancos del texto "X/Y" en la cabecera
    b = header_patch[:, :, 0]
    g = header_patch[:, :, 1]
    r = header_patch[:, :, 2]
    white_pixels = int(np.sum((r > 180) & (g > 180) & (b > 180)))

    if white_pixels < 6 and total_sidebar_commanders == 0:
        # No hay barra lateral desplegada (0 marchas en el mapa)
        return True, 0, _KNOWN_MAX_MARCHES

    # Guardar recorte limpio en test.png y usar ExtractTextAction / pytesseract
    try:
        cv2.imwrite("test.png", header_patch)
        extractor = ExtractTextAction(image_path="test.png", description="marchcount", delay=0)
        img_prep = extractor.preprocess_image("test.png")
        import pytesseract
        raw_text = pytesseract.image_to_string(
            img_prep,
            lang="eng",
            config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/",
        )
        digits = [int(ch) for ch in raw_text if ch.isdigit()]
        if len(digits) >= 2:
            cur_m = digits[0]
            max_m = digits[-1]
            if 1 <= max_m <= 7 and 0 <= cur_m <= max_m:
                _KNOWN_MAX_MARCHES = max(_KNOWN_MAX_MARCHES, max_m)
                # Si en la barra lateral vemos claramente N comandantes (ej. 5), aseguramos coherencia
                if total_sidebar_commanders > cur_m and total_sidebar_commanders <= max_m:
                    cur_m = total_sidebar_commanders
                return (cur_m < max_m), cur_m, max_m
    except Exception:
        pass

    # Respaldo directo por conteo de iconos en la barra lateral
    cur_m = total_sidebar_commanders
    max_m = _KNOWN_MAX_MARCHES
    return (cur_m < max_m), cur_m, max_m


class CheckGemMarchAvailableAction(Action):
    """
    Verifica si podemos buscar/enviar una marcha a gemas respetando los 3 tipos de íconos:
    - Pasos verdes (marching): en camino a las gemas -> ocupada.
    - Pico verde (gathering): recolectando gemas -> ocupada.
    - Flecha naranja (returning):
      * Si tiene espacio (returning_free): libre para mandar -> retorna True.
      * Si ya no puede cargar más gemas (returning_full): la deja regresar al castillo y revisa el contador X/Y.
        - Si el contador está en 5/5 (sin otra tropa en retorno libre): espera (retorna False).
        - Cuando la tropa llega al castillo y pasa a 4/5: retorna True para mandar tropa nueva.
    """

    def __init__(self, delay=0.05, retard=0):
        self.delay = delay
        self.retard = retard
        self.image = "check_march"
        self.window_title = "Rise of Kingdoms"
        self.window_handler = WindowHandler()
        self.image_finder = ImageFinder(threshold=0.76)

    def execute(self):
        screenshot, win = self.window_handler.screenshot_window(self.window_title)
        if screenshot is None or win is None:
            return False

        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        base_scale = self.image_finder._get_scaling_factor(screenshot_cv)

        sidebar = _analyze_sidebar_commanders(screenshot_cv, base_scale)
        has_free_slot, cur_m, max_m = _read_march_counter(screenshot_cv, sidebar["total_commanders"])

        n_marching = len(sidebar["marching"])
        n_gathering = len(sidebar["gathering"])
        n_ret_free = len(sidebar["returning_free"])
        n_ret_full = len(sidebar["returning_full"])

        print(
            colored(
                f"[GemMarch] Cabecera: {cur_m}/{max_m} | Pasos verdes (en camino): {n_marching} | "
                f"Pico verde (recolectando): {n_gathering} | Retorno libre (naranja): {n_ret_free} | "
                f"Retorno llena (al castillo): {n_ret_full}",
                "cyan",
            )
        )

        # Permitimos que el bot localice el siguiente módulo de gemas y, si las 5/5 marchas siguen
        # ocupadas al abrirlo, se quede esperando ahí mismo frente al módulo hasta que se libere una marcha.
        return True


class DispatchGemMarchAction(Action):
    """
    Despacha la marcha hacia el módulo de gemas tras pulsar GATHER:
    1) Prioridad 1: Tropas con ícono naranja de retorno (returning_free):
       - Ignora por completo las tropas con pasos verdes (en camino) o pico verde (recolectando).
       - Si al seleccionar una tropa en retorno aparece el botón March amarillo, la envía al módulo.
       - Si aparece el botón March en GRIS (ya no puede cargar más gemas), marca esa tropa como llena
         para dejarla regresar al castillo y prueba INMEDIATAMENTE si hay OTRA tropa regresando con espacio
         para que se ocupe de ese módulo.
    2) Prioridad 2: Si el botón New Troop está disponible (ej. 4/5), asigna 10,000 tropas y envía.
    3) Prioridad 3 (Quedarse esperando en el módulo hasta que se libere una marcha):
       - Si está en 5/5 y no hay marcha disponible, NO abandona el módulo ni sigue buscando:
         cierra el aviso gris y se queda esperando en este mismo módulo hasta que:
         a) Otra tropa termine de recolectar y aparezca con ícono naranja y espacio disponible, O
         b) La tropa llena llegue al castillo y el contador pase de 5/5 a 4/5 (apareciendo New Troop).
    """

    def __init__(self, OS_ROKBOT=None, troop_count=10000, delay=0.35, retard=0.4):
        self.OS_ROKBOT = OS_ROKBOT
        self.troop_count = str(troop_count)
        self.delay = delay
        self.retard = retard
        self.image = "dispatch_gem"
        self.window_title = "Rise of Kingdoms"
        self.window_handler = WindowHandler()
        self.image_finder = ImageFinder(threshold=0.76)

    def _get_bot(self):
        if self.OS_ROKBOT is not None:
            return self.OS_ROKBOT
        ui = GlobalVars().UI
        if ui is not None and hasattr(ui, "OS_ROKBOT"):
            return ui.OS_ROKBOT
        return None

    def _is_stopped(self):
        bot = self._get_bot()
        return bot is not None and bot.stop_event.is_set()

    def _wait_if_paused(self):
        bot = self._get_bot()
        while bot is not None and bot.pause_event.is_set() and not bot.stop_event.is_set():
            time.sleep(0.2)

    def _sleep_interruptible(self, duration):
        end_time = time.time() + duration
        while time.time() < end_time:
            if self._is_stopped():
                return False
            self._wait_if_paused()
            time.sleep(min(0.1, max(0.0, end_time - time.time())))
        return True

    def _click_screen(self, screen_x, screen_y):
        self.window_handler.click_screen(screen_x, screen_y, self.window_title)

    def _is_grey_march_visible(self, screenshot_cv, base_scale):
        grey_path = _resolve_media_path("Media/grey_march.png")
        grey_tpl = cv2.imread(grey_path)
        if grey_tpl is None:
            return False

        grey_top = grey_tpl[: int(grey_tpl.shape[0] * 0.62), :]

        for mult in [1.0, 0.92, 1.08]:
            fx = base_scale[0] * mult
            fy = base_scale[1] * mult
            nw = int(grey_top.shape[1] * fx)
            nh = int(grey_top.shape[0] * fy)
            if nw < 12 or nh < 10:
                continue
            resized = cv2.resize(grey_top, (nw, nh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(screenshot_cv, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= 0.72:
                patch = screenshot_cv[max_loc[1] : max_loc[1] + nh, max_loc[0] : max_loc[0] + nw]
                if patch.size > 0:
                    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                    if float(np.mean(hsv[:, :, 1])) < 55.0:
                        return True
        return False

    def _set_battering_ram_count(self, screenshot_cv, win, base_scale):
        input_x = None
        input_y = None

        ram_path = _resolve_media_path("Media/battering_ram.png")
        ram_tpl = cv2.imread(ram_path)
        if ram_tpl is not None:
            best_val = -1.0
            best_loc = None
            best_scale = base_scale
            best_size = (ram_tpl.shape[1], ram_tpl.shape[0])

            for mult in [1.0, 0.92, 1.08, 0.85, 1.15]:
                fx = base_scale[0] * mult
                fy = base_scale[1] * mult
                nw = int(ram_tpl.shape[1] * fx)
                nh = int(ram_tpl.shape[0] * fy)
                if nw < 12 or nh < 12:
                    continue
                resized = cv2.resize(ram_tpl, (nw, nh), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(screenshot_cv, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_val:
                    best_val = float(max_val)
                    best_loc = max_loc
                    best_scale = (fx, fy)
                    best_size = (nw, nh)

            if best_val >= 0.70 and best_loc is not None:
                ram_cx = best_loc[0] + best_size[0] // 2
                ram_cy = best_loc[1] + best_size[1] // 2
                input_x = int(win.left + ram_cx + 274 * best_scale[0])
                input_y = int(win.top + ram_cy + 10 * best_scale[1])

        if input_x is None or input_y is None:
            input_x = int(win.left + win.width * 0.715)
            input_y = int(win.top + win.height * 0.305)

        print(colored(f"[GemMarch] Asignando {self.troop_count} tropas en ({input_x}, {input_y})...", "cyan"))
        self._click_screen(input_x, input_y)
        time.sleep(0.25)

        self.window_handler.press_key("backspace", presses=7, interval=0.02, title=self.window_title)
        self.window_handler.press_key("delete", presses=5, interval=0.02, title=self.window_title)
        self.window_handler.write_text(self.troop_count, interval=0.03, title=self.window_title)
        time.sleep(0.08)
        self.window_handler.press_key("enter", presses=1, interval=0.02, title=self.window_title)
        time.sleep(0.25)
        return True

    def _reopen_gem_gather(self, win):
        """
        Vuelve a hacer clic en la coordenada exacta donde se encontró el módulo de gemas actual
        (_LAST_GEM_CLICK_POS) y pulsa el botón GATHER.
        """
        gather_path = _resolve_media_path("Media/gatheraction.png")

        # 1. Si GATHER ya está visible en pantalla, pulsarlo directamente
        ss_check, win_check = self.window_handler.screenshot_window(self.window_title)
        if ss_check is not None and win_check is not None:
            if self.image_finder.find_and_click_image(gather_path, ss_check, win_check, 0, 0, 0):
                self._sleep_interruptible(0.55)
                return True

        # 2. Probar en la coordenada exacta donde se abrió esta mina (_LAST_GEM_CLICK_POS)
        target_points = []
        if _LAST_GEM_CLICK_POS is not None:
            gx, gy = _LAST_GEM_CLICK_POS
            target_points.append((gx, gy))
            target_points.append((gx, gy + 6))

        center_x = int(win.left + win.width * 0.50)
        center_y = int(win.top + win.height * 0.50)
        target_points.append((center_x, center_y))

        for tx, ty in target_points:
            self._click_screen(tx, ty)
            if not self._sleep_interruptible(0.65):
                return False

            ss_g, win_g = self.window_handler.screenshot_window(self.window_title)
            if ss_g is None or win_g is None:
                return False

            if self.image_finder.find_and_click_image(gather_path, ss_g, win_g, 0, 0, 0):
                self._sleep_interruptible(0.55)
                return True

        return False

    def _try_dispatch_new_troop(self, base_scale):
        """
        Intenta hacer clic en New Troop, asignar 10,000 arietes y pulsar MARCH.
        """
        screenshot, win = self.window_handler.screenshot_window(self.window_title)
        if screenshot is None or win is None:
            return False

        if not self.image_finder.find_and_click_image(
            _resolve_media_path("Media/newtroopaction.png"), screenshot, win, 0, 0, 0
        ):
            return False

        if not self._sleep_interruptible(0.60):
            return False

        screenshot_nt, win_nt = self.window_handler.screenshot_window(self.window_title)
        if screenshot_nt is None or win_nt is None:
            return False

        screenshot_nt_cv = cv2.cvtColor(np.array(screenshot_nt), cv2.COLOR_RGB2BGR)
        self._set_battering_ram_count(screenshot_nt_cv, win_nt, base_scale)

        screenshot_march, win_march = self.window_handler.screenshot_window(self.window_title)
        if screenshot_march is not None and win_march is not None:
            if self.image_finder.find_and_click_image(
                _resolve_media_path("Media/marchaction.png"), screenshot_march, win_march, 0, 0, 0
            ):
                print(colored(f"[GemMarch] Nueva marcha enviada desde el castillo con {self.troop_count} tropas.", "green"))
                return True
        return False

    def _try_dispatch_returning_commanders(self, screenshot_cv, win, base_scale, returning_free_list):
        """
        Recorre las tropas con ícono naranja de retorno disponibles.
        Si la primera está llena (botón March gris), la marca como llena y prueba de inmediato
        si hay otra tropa en retorno con espacio para que se ocupe de este módulo.
        Retorna: (sent_successfully: bool, clicked_any: bool)
        """
        clicked_any = False
        for cx_rel, cy_rel, sc, tpl_name in returning_free_list:
            if self._is_stopped():
                return False, clicked_any
            self._wait_if_paused()

            cmd_x = int(win.left + cx_rel)
            cmd_y = int(win.top + cy_rel)
            print(
                colored(
                    f"[GemMarch] Probando tropa con ícono naranja de retorno ({sc:.2f}) en ({cmd_x}, {cmd_y})...",
                    "cyan",
                )
            )
            self._click_screen(cmd_x, cmd_y)
            clicked_any = True
            if not self._sleep_interruptible(0.45):
                return False, clicked_any

            screenshot2, win2 = self.window_handler.screenshot_window(self.window_title)
            if screenshot2 is None or win2 is None:
                continue

            screenshot2_cv = cv2.cvtColor(np.array(screenshot2), cv2.COLOR_RGB2BGR)

            if self._is_grey_march_visible(screenshot2_cv, base_scale):
                print(
                    colored(
                        "[GemMarch] La tropa en retorno ya no puede cargar más gemas (March gris). "
                        "Dejándola regresar al castillo y verificando si hay otra tropa libre...",
                        "yellow",
                    )
                )
                _mark_commander_as_full(screenshot_cv, cx_rel, cy_rel, base_scale)
                continue

            if self.image_finder.find_and_click_image(
                _resolve_media_path("Media/smallmarchaction.png"), screenshot2, win2, 0, 0, 0
            ):
                print(colored("[GemMarch] Tropa en retorno con espacio asignada exitosamente al módulo de gemas.", "green"))
                return True, clicked_any

            if self.image_finder.find_and_click_image(
                _resolve_media_path("Media/marchaction.png"), screenshot2, win2, 0, 0, 0
            ):
                print(colored("[GemMarch] Tropa en retorno enviada exitosamente al módulo de gemas.", "green"))
                return True, clicked_any

            _mark_commander_as_full(screenshot_cv, cx_rel, cy_rel, base_scale)

        return False, clicked_any

    def execute(self):
        screenshot, win = self.window_handler.screenshot_window(self.window_title)
        if screenshot is None or win is None:
            return False

        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        base_scale = self.image_finder._get_scaling_factor(screenshot_cv)

        had_new_troop_initially = self.image_finder.find_image(
            _resolve_media_path("Media/newtroopaction.png"), screenshot
        )

        sidebar = _analyze_sidebar_commanders(screenshot_cv, base_scale)
        returning_free_list = sidebar["returning_free"]

        # 1. Intentar con las tropas que tienen ícono naranja de retorno (y si una está llena, probar la siguiente)
        sent_ok, clicked_any_returning = self._try_dispatch_returning_commanders(
            screenshot_cv, win, base_scale, returning_free_list
        )
        if sent_ok:
            return True

        # 2. Si había botón New Troop al abrir GATHER, intentar enviar tropa nueva desde el castillo
        if had_new_troop_initially:
            if clicked_any_returning:
                neutral_x = int(win.left + win.width * 0.50)
                neutral_y = int(win.top + win.height * 0.18)
                self._click_screen(neutral_x, neutral_y)
                self._sleep_interruptible(0.25)
                self._reopen_gem_gather(win)

            if self._try_dispatch_new_troop(base_scale):
                return True

        # 3. No hay marcha libre en este instante (estamos en 5/5 y cualquier tropa en retorno está llena):
        #    Cerramos el recuadro para dejar regresar las tropas llenas al castillo y NOS QUEDAMOS ESPERANDO
        #    aquí mismo en este módulo de gemas sin seguir buscando hasta que se libere una marcha.
        neutral_x = int(win.left + win.width * 0.50)
        neutral_y = int(win.top + win.height * 0.18)
        self._click_screen(neutral_x, neutral_y)
        self._sleep_interruptible(0.30)

        print(
            colored(
                "[GemMarch] Módulo de gemas localizado y marchas ocupadas (5/5). "
                "Esperando en este mismo módulo a que se libere una marcha (4/5) o una tropa en retorno tenga espacio...",
                "yellow",
            )
        )

        wait_start = time.time()
        max_wait_seconds = 600  # Hasta 10 minutos de espera frente al módulo
        reopen_failures = 0

        while time.time() - wait_start < max_wait_seconds:
            if not self._sleep_interruptible(2.0):
                return False

            ss_wait, win_wait = self.window_handler.screenshot_window(self.window_title)
            if ss_wait is None or win_wait is None:
                continue

            ss_wait_cv = cv2.cvtColor(np.array(ss_wait), cv2.COLOR_RGB2BGR)
            sidebar_w = _analyze_sidebar_commanders(ss_wait_cv, base_scale)
            free_slot_w, cur_w, max_w = _read_march_counter(ss_wait_cv, sidebar_w["total_commanders"])

            n_ret_free = len(sidebar_w["returning_free"])

            if GlobalVars().UI is not None:
                try:
                    GlobalVars().UI.currentState(f"En módulo esperando\n{cur_w}/{max_w} -> {max_w - 1}/{max_w}")
                except Exception:
                    pass

            # Solo intentamos abrir el módulo cuando:
            # A) Aparece una tropa con flecha naranja de retorno no marcada como llena, O
            # B) El contador en la cabeza bajó de 5/5 a 4/5 (free_slot_w == True)
            if n_ret_free == 0 and not free_slot_w:
                continue

            if n_ret_free > 0:
                print(
                    colored(
                        "[GemMarch] ¡Detectada tropa en retorno (flecha naranja)! Abriendo módulo para verificar espacio...",
                        "cyan",
                    )
                )
            else:
                print(
                    colored(
                        f"[GemMarch] ¡Marcha liberada en el castillo ({cur_w}/{max_w})! Abriendo módulo para enviar tropa nueva...",
                        "green",
                    )
                )

            if not self._reopen_gem_gather(win_wait):
                reopen_failures += 1
                if reopen_failures >= 4:
                    print(
                        colored(
                            "[GemMarch] El módulo de gemas ya no está disponible (fue recolectado por otro jugador). Reanudando búsqueda...",
                            "yellow",
                        )
                    )
                    return False
                continue

            reopen_failures = 0

            ss_re, win_re = self.window_handler.screenshot_window(self.window_title)
            if ss_re is None or win_re is None:
                continue
            ss_re_cv = cv2.cvtColor(np.array(ss_re), cv2.COLOR_RGB2BGR)
            sidebar_re = _analyze_sidebar_commanders(ss_re_cv, base_scale)

            # Si hay tropa en retorno libre, probarla primero
            if len(sidebar_re["returning_free"]) > 0:
                sent_ret, _ = self._try_dispatch_returning_commanders(
                    ss_re_cv, win_re, base_scale, sidebar_re["returning_free"]
                )
                if sent_ret:
                    return True

            # Si el botón New Troop está disponible (porque ya llegó al castillo y pasó a 4/5), enviar tropa nueva
            if self._try_dispatch_new_troop(base_scale):
                return True

            # Si aún no estaba disponible New Troop o la tropa en retorno también estaba llena,
            # cerramos el popup y SEGUIMOS ESPERANDO en este mismo módulo (sin retornar False).
            self._click_screen(int(win_re.left + win_re.width * 0.50), int(win_re.top + win_re.height * 0.18))
            self._sleep_interruptible(0.30)

        return False


class FindGemAction(Action):
    GEM_TEMPLATES = [
        ("Media/gemdepo.png", [0.90, 1.0, 1.12]),
        ("Media/gemdepo2.png", [0.80, 0.92, 1.05]),
        ("Media/gem_mine_crystals.png", [0.36, 0.44, 0.54, 0.68]),
        ("Media/gem_mine_core.png", [0.34, 0.42, 0.52, 0.65]),
    ]

    def __init__(
        self,
        OS_ROKBOT=None,
        threshold=0.56,
        max_legs=24,
        map_load_delay=0.45,
        delay=0.05,
        retard=0,
    ):
        self.OS_ROKBOT = OS_ROKBOT
        self.threshold = threshold
        self.max_legs = max_legs
        self.map_load_delay = map_load_delay
        self.delay = delay
        self.retard = retard
        self.image = "gemdepo"
        self.window_title = "Rise of Kingdoms"
        self.window_handler = WindowHandler()
        self.image_finder = ImageFinder(threshold=0.75)

        self.current_leg = 1
        self.current_step = 0
        self.need_post_dispatch_escape = False
        self.excluded_screen_points = []
        self.pending_sibling_vectors = []

        self.loaded_templates = self._load_templates()
        self._cached_scale_key = None
        self._cached_resized_templates = []

    def _load_templates(self):
        loaded = []
        for rel_path, scales in self.GEM_TEMPLATES:
            full_path = _resolve_media_path(rel_path)
            img = cv2.imread(full_path)
            if img is None:
                continue

            if "gem_mine_core" in rel_path:
                h, w = img.shape[:2]
                img = img[int(h * 0.12) : int(h * 0.88), int(w * 0.14) : int(w * 0.88)]

            loaded.append((rel_path, img, scales))
        return loaded

    def _get_resized_templates(self, base_scale_x, base_scale_y):
        scale_key = (round(base_scale_x, 3), round(base_scale_y, 3))
        if self._cached_scale_key == scale_key and self._cached_resized_templates:
            return self._cached_resized_templates

        prepared = []
        for name, template_img, scale_multipliers in self.loaded_templates:
            for mult in scale_multipliers:
                fx = base_scale_x * mult
                fy = base_scale_y * mult
                new_w = int(template_img.shape[1] * fx)
                new_h = int(template_img.shape[0] * fy)
                if new_w < 16 or new_h < 16:
                    continue
                resized = cv2.resize(template_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                prepared.append((name, resized, new_w, new_h))

        self._cached_scale_key = scale_key
        self._cached_resized_templates = prepared
        return prepared

    def reset_spiral(self):
        self.current_leg = 1
        self.current_step = 0
        self.need_post_dispatch_escape = False
        self.excluded_screen_points = []
        self.pending_sibling_vectors = []

    def _get_bot(self):
        if self.OS_ROKBOT is not None:
            return self.OS_ROKBOT
        ui = GlobalVars().UI
        if ui is not None and hasattr(ui, "OS_ROKBOT"):
            return ui.OS_ROKBOT
        return None

    def _is_stopped(self):
        bot = self._get_bot()
        return bot is not None and bot.stop_event.is_set()

    def _wait_if_paused(self):
        bot = self._get_bot()
        while bot is not None and bot.pause_event.is_set() and not bot.stop_event.is_set():
            time.sleep(0.2)

    def _sleep_interruptible(self, duration):
        end_time = time.time() + duration
        while time.time() < end_time:
            if self._is_stopped():
                return False
            self._wait_if_paused()
            time.sleep(min(0.08, max(0.0, end_time - time.time())))
        return True

    def _get_direction_for_leg(self, leg):
        if leg % 4 == 1:
            return "left", 0.36
        elif leg % 4 == 2:
            return "down", 0.28
        elif leg % 4 == 3:
            return "right", 0.36
        else:
            return "up", 0.28

    def _move_camera(self, key, key_hold_time):
        self.window_handler.hold_key(key, duration=key_hold_time, title=self.window_title)
        self._sleep_interruptible(self.map_load_delay)

    def _move_towards_vector(self, dx, dy):
        if abs(dx) > 120:
            horiz_key = "right" if dx > 0 else "left"
            hold_x = min(0.32, max(0.14, abs(dx) / 1400.0))
            self._move_camera(horiz_key, hold_x)
        if abs(dy) > 90:
            vert_key = "down" if dy > 0 else "up"
            hold_y = min(0.26, max(0.12, abs(dy) / 1000.0))
            self._move_camera(vert_key, hold_y)

    def _validate_gem_colors(self, bgr_patch):
        if bgr_patch is None or bgr_patch.size == 0:
            return False, 0.0, 0.0, 0.0

        hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        total_pixels = float(bgr_patch.shape[0] * bgr_patch.shape[1])

        red_mask1 = cv2.inRange(hsv, np.array([0, 75, 60]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([168, 75, 60]), np.array([180, 255, 255]))
        red_ratio = cv2.countNonZero(cv2.bitwise_or(red_mask1, red_mask2)) / total_pixels

        sand_mask = cv2.inRange(hsv, np.array([10, 40, 70]), np.array([28, 240, 255]))
        sand_ratio = cv2.countNonZero(sand_mask) / total_pixels

        is_valid = (red_ratio >= 0.022 and sand_ratio >= 0.10) or (red_ratio >= 0.045 and sand_ratio >= 0.06)
        bonus = min(0.15, red_ratio * 0.8 + sand_ratio * 0.15) if is_valid else 0.0
        return is_valid, bonus, red_ratio, sand_ratio

    def _is_point_excluded(self, cx, cy, w_full, h_full, exclude_points=None, exclude_corridor_dir=None):
        """
        Verifica si el candidato (cx, cy) corresponde al mismo módulo de gemas al que acabamos de enviar tropa:
        1) Por proximidad directa a cualquiera de los puntos recién clicados (radio 130x115 px).
        2) Por el corredor de arrastre desde cada punto previo (tanto la coordenada original como el centro 50%,50%)
           en la dirección opuesta al movimiento de la cámara:
           - Si la cámara se movió a 'left', el módulo anterior se desplazó hacia la derecha (cx >= sx - 45, |cy - sy| <= 110).
           - Si la cámara se movió a 'right', el módulo anterior se desplazó hacia la izquierda (cx <= sx + 45, |cy - sy| <= 110).
           - Si la cámara se movió a 'down', el módulo anterior se desplazó hacia arriba (cy <= sy + 45, |cx - sx| <= 125).
           - Si la cámara se movió a 'up', el módulo anterior se desplazó hacia abajo (cy >= sy - 45, |cx - sx| <= 125).
        """
        anchors = list(exclude_points) if exclude_points else []
        if exclude_corridor_dir is not None and not anchors:
            anchors.append((w_full * 0.50, h_full * 0.50))

        for sx, sy in anchors:
            if abs(cx - sx) < 130 and abs(cy - sy) < 115:
                return True

            if exclude_corridor_dir == "left":
                if cx >= sx - 45 and abs(cy - sy) <= 110:
                    return True
            elif exclude_corridor_dir == "right":
                if cx <= sx + 45 and abs(cy - sy) <= 110:
                    return True
            elif exclude_corridor_dir == "down":
                if cy <= sy + 45 and abs(cx - sx) <= 125:
                    return True
            elif exclude_corridor_dir == "up":
                if cy >= sy - 45 and abs(cx - sx) <= 125:
                    return True

        return False

    def _find_candidates_in_screenshot(self, screenshot_cv, exclude_points=None, exclude_corridor_dir=None):
        h_full, w_full = screenshot_cv.shape[:2]

        x_min = int(w_full * 0.10)
        x_max = int(w_full * 0.88)
        y_min = int(h_full * 0.12)
        y_max = int(h_full * 0.82)
        roi = screenshot_cv[y_min:y_max, x_min:x_max]

        base_scale_x, base_scale_y = self.image_finder._get_scaling_factor(screenshot_cv)
        prepared_templates = self._get_resized_templates(base_scale_x, base_scale_y)

        raw_boxes = []
        raw_scores = []
        raw_meta = []

        for name, resized, new_w, new_h in prepared_templates:
            if new_w >= roi.shape[1] or new_h >= roi.shape[0]:
                continue

            result = cv2.matchTemplate(roi, resized, cv2.TM_CCOEFF_NORMED)
            res_work = result.copy()

            for _ in range(4):
                _, max_val, _, max_loc = cv2.minMaxLoc(res_work)
                if max_val < self.threshold:
                    break

                pt_x, pt_y = max_loc
                sup_x1 = max(0, pt_x - 45)
                sup_y1 = max(0, pt_y - 45)
                sup_x2 = min(res_work.shape[1], pt_x + 45)
                sup_y2 = min(res_work.shape[0], pt_y + 45)
                res_work[sup_y1:sup_y2, sup_x1:sup_x2] = 0.0

                abs_x1 = x_min + pt_x
                abs_y1 = y_min + pt_y
                cx = abs_x1 + new_w // 2
                cy = abs_y1 + new_h // 2

                if self._is_point_excluded(cx, cy, w_full, h_full, exclude_points, exclude_corridor_dir):
                    continue

                score = float(max_val)
                patch = roi[pt_y : pt_y + new_h, pt_x : pt_x + new_w]
                color_ok, bonus, red_r, sand_r = self._validate_gem_colors(patch)

                if color_ok or score >= 0.74:
                    final_score = score + bonus
                    raw_boxes.append([abs_x1, abs_y1, abs_x1 + new_w, abs_y1 + new_h])
                    raw_scores.append(final_score)
                    raw_meta.append((name, score, final_score, red_r, sand_r))

        if not raw_boxes:
            return []

        order = np.argsort(raw_scores)[::-1]
        selected_candidates = []

        for idx in order:
            box = raw_boxes[idx]
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2

            too_close = False
            for prev_box, _ in selected_candidates:
                pcx = (prev_box[0] + prev_box[2]) // 2
                pcy = (prev_box[1] + prev_box[3]) // 2
                if abs(cx - pcx) < 75 and abs(cy - pcy) < 75:
                    too_close = True
                    break

            if not too_close:
                selected_candidates.append((box, raw_meta[idx]))
            if len(selected_candidates) >= 4:
                break

        return selected_candidates

    def _perform_map_click(self, screen_x, screen_y):
        try:
            self.window_handler.click_screen(screen_x, screen_y, self.window_title)
        except Exception as e:
            print(f"[FindGem] Aviso al hacer clic en ({screen_x}, {screen_y}): {e}")

    def _is_gather_button_visible(self):
        screenshot, _ = self.window_handler.screenshot_window(self.window_title)
        if screenshot is None:
            return False
        gather_path = _resolve_media_path("Media/gatheraction.png")
        return self.image_finder.find_image(gather_path, screenshot)

    def _scan_and_click_gem(self, exclude_points=None, exclude_corridor_dir=None, record_siblings=True):
        screenshot, win = self.window_handler.screenshot_window(self.window_title)
        if screenshot is None or win is None:
            return False

        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        candidates = self._find_candidates_in_screenshot(
            screenshot_cv,
            exclude_points=exclude_points,
            exclude_corridor_dir=exclude_corridor_dir,
        )

        if not candidates:
            return False

        if record_siblings and len(candidates) >= 2:
            print(colored(f"[FindGem] ¡{len(candidates)} minas de gemas detectadas en la misma pantalla!", "cyan"))

        for i, (box, meta) in enumerate(candidates):
            if self._is_stopped():
                return False
            self._wait_if_paused()

            name, raw_sc, final_sc, red_r, sand_r = meta
            w_box = box[2] - box[0]
            h_box = box[3] - box[1]
            cx_rel = box[0] + w_box * 0.50
            cy_rel = box[1] + h_box * 0.56
            click_x = int(win.left + cx_rel)
            click_y = int(win.top + cy_rel)

            print(
                colored(
                    f"[FindGem] Mina #{i + 1}/{len(candidates)} ({name}) score={raw_sc:.2f} -> clic en ({click_x}, {click_y})",
                    "green",
                )
            )

            self._perform_map_click(click_x, click_y)

            if not self._sleep_interruptible(0.65):
                return False

            gather_ok = self._is_gather_button_visible()
            if not gather_ok:
                if not self._sleep_interruptible(0.35):
                    return False
                gather_ok = self._is_gather_button_visible()

            if not gather_ok:
                # Reintentar clic directamente sobre la misma mina (no en 50%,50% para no cerrar el popup)
                self._perform_map_click(click_x, click_y + 5)
                if not self._sleep_interruptible(0.65):
                    return False
                gather_ok = self._is_gather_button_visible()

            if gather_ok:
                global _LAST_GEM_CLICK_POS
                _LAST_GEM_CLICK_POS = (click_x, click_y)

                # Excluir tanto la coordenada original (cx_rel, cy_rel) como el centro (50%, 50%)
                # para que NUNCA vuelva a mandar otra marcha al mismo módulo.
                self.excluded_screen_points.append((cx_rel, cy_rel))
                self.excluded_screen_points.append((win.width * 0.50, win.height * 0.50))
                self.need_post_dispatch_escape = True

                if record_siblings:
                    self.pending_sibling_vectors = []
                    for j in range(i + 1, len(candidates)):
                        other_box, _ = candidates[j]
                        ox = (other_box[0] + other_box[2]) * 0.50
                        oy = (other_box[1] + other_box[3]) * 0.56
                        # Solo guardar si está claramente separada (> 95 px) de la primera mina
                        if abs(ox - cx_rel) > 95 or abs(oy - cy_rel) > 95:
                            self.pending_sibling_vectors.append((ox - cx_rel, oy - cy_rel))

                return True

            print(colored("[FindGem] Nodo ocupado o sin GATHER, probando siguiente mina en pantalla...", "yellow"))

        return False

    def execute(self):
        self.window_handler.activate_window(self.window_title)

        # 1. ÚNICAMENTE si en el escaneo anterior se detectaron 2 o más minas distintas en la misma pantalla,
        #    vamos por la 2da mina excluyendo la 1ra (tanto su posición original como el centro de pantalla).
        while self.pending_sibling_vectors:
            if self._is_stopped():
                return False
            self._wait_if_paused()

            dx, dy = self.pending_sibling_vectors.pop(0)
            print(
                colored(
                    f"[FindGem] Buscando la 2da mina detectada en esta zona (dx={dx:.0f}, dy={dy:.0f}) excluyendo la anterior...",
                    "cyan",
                )
            )
            # Primero probamos en la vista actual excluyendo los puntos ya clicados
            if self._scan_and_click_gem(
                exclude_points=self.excluded_screen_points,
                exclude_corridor_dir=None,
                record_siblings=False,
            ):
                return True

            # Si el centrado de cámara sacó a la 2da mina del borde, nos movemos hacia ella
            self._move_towards_vector(dx, dy)
            if self._scan_and_click_gem(
                exclude_points=self.excluded_screen_points,
                exclude_corridor_dir=None,
                record_siblings=False,
            ):
                return True

        # Si estamos al inicio absoluto de la espiral (sin haber enviado ninguna marcha aún)
        if self.current_leg == 1 and self.current_step == 0 and not self.need_post_dispatch_escape:
            if self._scan_and_click_gem(exclude_points=None, exclude_corridor_dir=None, record_siblings=True):
                return True

        # 2. Avanzar en la espiral alejándonos del módulo anterior
        for duration in range(self.current_leg, self.max_legs):
            key, key_hold = self._get_direction_for_leg(duration)
            start_step = self.current_step if duration == self.current_leg else 0

            for step in range(start_step, duration):
                if self._is_stopped():
                    return False
                self._wait_if_paused()

                if GlobalVars().UI is not None:
                    try:
                        GlobalVars().UI.currentState(f"FindGem\n{key} ({step + 1}/{duration})")
                    except Exception:
                        pass

                # Si acabamos de enviar una tropa a un módulo, damos un desplazamiento más largo
                # para sacar ese módulo de la pantalla y activamos la exclusión de su corredor de arrastre.
                corridor_dir = None
                corridor_anchors = None
                actual_hold = key_hold
                if self.need_post_dispatch_escape:
                    actual_hold = 0.68 if key in ("left", "right") else 0.52
                    corridor_dir = key
                    corridor_anchors = list(self.excluded_screen_points)
                    self.need_post_dispatch_escape = False
                    self.excluded_screen_points = []

                self._move_camera(key, actual_hold)

                if self._scan_and_click_gem(
                    exclude_points=corridor_anchors,
                    exclude_corridor_dir=corridor_dir,
                    record_siblings=True,
                ):
                    if step + 1 < duration:
                        self.current_leg = duration
                        self.current_step = step + 1
                    else:
                        self.current_leg = duration + 1
                        self.current_step = 0
                    return True

            self.current_step = 0

        print(colored("[FindGem] Espiral completada. Reiniciando desde la ciudad...", "yellow"))
        self.reset_spiral()
        return False
