import os
import time
import cv2
import numpy as np
import pyautogui
from termcolor import colored


class ImageFinder:
    def __init__(self, threshold=0.78):
        self.threshold = threshold
        self.template_resolution = (1280, 720)  # original resolution at which the template was taken

    def _get_scaling_factor(self, screenshot):
        screen_resolution = (screenshot.shape[1] - 8, screenshot.shape[0] - 31)  # (width, height)

        scaling_factor = (
            screen_resolution[0] / self.template_resolution[0],
            screen_resolution[1] / self.template_resolution[1],
        )  # (scale_x, scale_y)
        return scaling_factor

    def _match_image(self, target_image_path, screenshot):
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        target_image = cv2.imread(target_image_path)

        base_scale = self._get_scaling_factor(screenshot_cv)

        # Candidatos de imagen (incluye respaldo gatheraction_old.png y recorte interno de searchaction.png)
        candidate_images = [(target_image, [1.0, 0.93, 1.07, 0.86, 1.14, 0.80, 1.22])]
        if "gatheraction.png" in target_image_path and os.path.exists("Media/gatheraction_old.png"):
            old_img = cv2.imread("Media/gatheraction_old.png")
            if old_img is not None:
                candidate_images.append((old_img, [1.0, 0.93, 1.07, 1.14]))
        if "searchaction.png" in target_image_path and target_image is not None:
            th, tw = target_image.shape[:2]
            if th > 14 and tw > 20:
                cropped_search = target_image[5 : th - 5, 7 : tw - 7]
                candidate_images.insert(0, (cropped_search, [1.0, 0.93, 1.07, 0.86, 1.14, 0.80, 1.22]))

        best_max_val = -1.0
        best_result = None
        best_resized = None
        best_target = target_image
        best_scale = base_scale

        for cand_img, mult_list in candidate_images:
            if cand_img is None:
                continue
            for mult in mult_list:
                eff_scale = (base_scale[0] * mult, base_scale[1] * mult)
                resized_img = cv2.resize(
                    cand_img,
                    None,
                    fx=eff_scale[0],
                    fy=eff_scale[1],
                    interpolation=cv2.INTER_AREA,
                )
                if (
                    resized_img.shape[0] >= screenshot_cv.shape[0]
                    or resized_img.shape[1] >= screenshot_cv.shape[1]
                    or resized_img.shape[0] < 8
                    or resized_img.shape[1] < 8
                ):
                    continue

                result = cv2.matchTemplate(screenshot_cv, resized_img, cv2.TM_CCOEFF_NORMED)
                _, cur_max, _, cur_loc = cv2.minMaxLoc(result)
                cur_max = float(cur_max)

                # Para searchaction.png: aceptar coincidencia desde 0.56 si el botón tiene el color celeste de SEARCH
                if "searchaction.png" in target_image_path and 0.56 <= cur_max < self.threshold:
                    if cur_loc[1] >= int(screenshot_cv.shape[0] * 0.40):
                        rh, rw = resized_img.shape[:2]
                        patch = screenshot_cv[cur_loc[1] : cur_loc[1] + rh, cur_loc[0] : cur_loc[0] + rw]
                        if patch.size > 0:
                            hsv_p = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                            cyan_mask = cv2.inRange(hsv_p, np.array([88, 130, 160]), np.array([112, 255, 255]))
                            cyan_ratio = float(cv2.countNonZero(cyan_mask)) / float(rh * rw)
                            if cyan_ratio >= 0.38:
                                cur_max = max(cur_max, self.threshold + 0.02)
                                result[cur_loc[1], cur_loc[0]] = cur_max

                # Evitar falso positivo cuando el botón March está deshabilitado en gris (sin saturación amarilla)
                if cur_max >= self.threshold and ("smallmarchaction" in target_image_path or "marchaction" in target_image_path):
                    rh, rw = resized_img.shape[:2]
                    patch = screenshot_cv[cur_loc[1]:cur_loc[1] + rh, cur_loc[0]:cur_loc[0] + rw]
                    if patch.size > 0:
                        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                        mean_sat = float(np.mean(hsv_patch[:, :, 1]))
                        if mean_sat < 65.0:
                            cur_max = 0.0
                            result[:] = 0.0

                if cur_max > best_max_val:
                    best_max_val = cur_max
                    best_result = result
                    best_resized = resized_img
                    best_target = cand_img
                    best_scale = eff_scale

                # Si la escala 1.0 ya supera ampliamente el umbral, no hace falta probar más escalas
                if mult == 1.0 and cur_max >= self.threshold + 0.04:
                    break
            if best_max_val >= self.threshold + 0.04:
                break

        # Respaldo directo por detección del botón celeste SEARCH en el popup de recursos si cambió la tipografía
        if "searchaction.png" in target_image_path and best_max_val < self.threshold:
            sh, sw = screenshot_cv.shape[:2]
            rx1, rx2 = int(sw * 0.12), int(sw * 0.88)
            ry1, ry2 = int(sh * 0.44), int(sh * 0.82)
            roi = screenshot_cv[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                cyan_btn_mask = cv2.inRange(hsv_roi, np.array([90, 155, 175]), np.array([108, 255, 255]))
                contours, _ = cv2.findContours(cyan_btn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    aspect = float(bw) / float(max(1, bh))
                    if 54 <= bw <= 165 and 20 <= bh <= 58 and 1.8 <= aspect <= 4.5:
                        fill_ratio = float(cv2.contourArea(cnt)) / float(bw * bh)
                        if fill_ratio >= 0.45:
                            abs_x = rx1 + bx
                            abs_y = ry1 + by
                            best_max_val = self.threshold + 0.03
                            best_scale = base_scale
                            best_target = np.zeros((bh, bw, 3), dtype=np.uint8)
                            best_resized = best_target
                            best_result = np.zeros((sh, sw), dtype=np.float32)
                            best_result[abs_y, abs_x] = best_max_val
                            break

        # find all matches above threshold
        locations = np.where(best_result >= self.threshold)
        matches = list(zip(*locations[::-1]))  # reverse x, y and make list of tuples

        # draw rectangles on all matched locations
        w, h = (best_resized.shape[1], best_resized.shape[0])
        for pt in matches:
            cv2.rectangle(screenshot_cv, pt, (int(pt[0] + w), int(pt[1] + h)), (255, 0, 255), 4)

        # delete screenshot.png
        try:
            os.remove("screenshot.png")
        except Exception:
            pass
        # save screenshot with rectangles
        cv2.imwrite("screenshot.png", screenshot_cv)
        # return the number of matches, screenshot with rectangles, and the highest matching value
        return best_scale, matches, len(matches), best_max_val, best_target, screenshot_cv

    def find_and_click_image(self, target_image_path, screenshot, win, x_offset, y_offset, max_matches):
        best_scale, best_loc, num_matches, best_max_val, target_image, screenshot_cv = self._match_image(
            target_image_path, screenshot
        )

        if best_max_val >= self.threshold:
            # Create an array of rectangles with format [[x1, y1, x2, y2], ...]
            rects = np.array(
                [
                    (
                        pt[0],
                        pt[1],
                        pt[0] + target_image.shape[1] * best_scale[0],
                        pt[1] + target_image.shape[0] * best_scale[1],
                    )
                    for pt in best_loc
                ]
            )

            # Apply non-maximum suppression to the rectangles
            pick = ImageFinder.non_max_suppression_fast(rects, 0.3)

            for (startX, startY, endX, endY) in pick:
                cv2.rectangle(screenshot_cv, (int(startX), int(startY)), (int(endX), int(endY)), (255, 0, 255), 2)

                center_x = int(startX + (endX - startX) // 2 + win.left)
                center_y = int(startY + (endY - startY) // 2 + win.top)

                # scale offsets according to the best scaling factor
                x_offset_scaled = int(x_offset * best_scale[0])
                y_offset_scaled = int(y_offset * best_scale[1])

            print(colored(f"found {target_image_path} {len(pick)}x at {best_max_val}%", "green"))
            if len(pick) >= max_matches and max_matches != 0:
                return False
            if len(pick) < max_matches and max_matches != 0:
                return True

            try:
                target_x = center_x + x_offset_scaled
                target_y = center_y + y_offset_scaled
                from window_handler import WindowHandler
                WindowHandler().click_screen(target_x, target_y)
                time.sleep(0.05)
            except Exception:
                print("woops")
            return True
        else:
            if target_image_path != "Media/captchachest.png":
                print(colored(f"No matches for {target_image_path} found in screenshot (best={best_max_val:.2f}).", "red"))
            if max_matches != 0:
                return True
            return False

    @staticmethod
    def non_max_suppression_fast(boxes, overlapThresh):
        if len(boxes) == 0:
            return []

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick]

    def find_image(self, target_image_path, screenshot):
        best_scale, best_loc, num_matches, best_max_val, target_image, screenshot_cv = self._match_image(
            target_image_path, screenshot
        )

        if best_max_val >= self.threshold:
            print(colored(f"found {target_image_path} {best_max_val}%", "green"))
            return True
        else:
            print(colored(f"No matches for {target_image_path} found in screenshot (best={best_max_val:.2f}).", "red"))
            return False
