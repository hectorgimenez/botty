from char.i_char import IChar
from config import Config
from logger import Logger
from pather import Location, Pather
from typing import Union
from item.pickit import PickIt
from template_finder import TemplateFinder
from town.town_manager import TownManager
from ui import UiManager
from utils.misc import wait
from dataclasses import dataclass
from screen import Screen
import random
import cv2
import numpy as np
from typing import Tuple


@dataclass
class Orientation:
    number: int
    monitor_position: Tuple[float, float]
    reversed: bool
    template_to_follow: str

    def __init__(self, screen: Screen, num: int = None, is_reversed: bool = False, template: str = None):
        orientations = {
            0: screen.convert_abs_to_monitor((300, 10)),
            1: screen.convert_abs_to_monitor((-100, 110)),
            2: screen.convert_abs_to_monitor((-300, -10)),
            3: screen.convert_abs_to_monitor((250, -240)),
        }
        template_by_orientation = {
            0: Mephisto.WALL_TOP_RIGHT,
            1: Mephisto.WALL_BOTTOM_RIGHT,
            2: Mephisto.WALL_BOTTOM_LEFT,
            3: Mephisto.WALL_TOP_LEFT
        }
        template_by_reversed_orientation = {
            0: Mephisto.WALL_BOTTOM_LEFT,
            1: Mephisto.WALL_TOP_LEFT,
            2: Mephisto.WALL_TOP_RIGHT,
            3: Mephisto.WALL_BOTTOM_RIGHT
        }

        if num is not None:
            self.number = num
            self.reversed = is_reversed
            if is_reversed:
                self.template_to_follow = template_by_reversed_orientation[num]
            else:
                self.template_to_follow = template_by_orientation[num]
        else:
            if is_reversed:
                number = template in template_by_reversed_orientation
            else:
                number = template in template_by_orientation
            self.number = number
            self.template_to_follow = template
            self.reversed = is_reversed

        coords = orientations[self.number]
        if is_reversed:
            coords = (coords[1] * -1, coords[0])
        self.monitor_position = coords

    @classmethod
    def create_by_template(cls, screen: Screen, template: str, is_reversed: bool):
        return Orientation(screen, None, is_reversed, template)

    @classmethod
    def create_by_number(cls, screen: Screen, number: int, is_reversed: bool):
        return Orientation(screen, number, is_reversed)


class Mephisto:
    WALL_BOTTOM_LEFT = ["MEPH_WALL_BOTTOM_LEFT_0", "MEPH_WALL_BOTTOM_LEFT_1"]
    WALL_BOTTOM_RIGHT = ["MEPH_WALL_BOTTOM_RIGHT_0", "MEPH_WALL_BOTTOM_RIGHT_1"]
    WALL_TOP_LEFT = ["MEPH_WALL_TOP_LEFT_0", "MEPH_WALL_TOP_LEFT_1", "MEPH_WALL_TOP_LEFT_2", "MEPH_WALL_TOP_LEFT_3"]
    WALL_TOP_RIGHT = ["MEPH_WALL_TOP_RIGHT_0", "MEPH_WALL_TOP_RIGHT_1", "MEPH_WALL_TOP_RIGHT_2",
                      "MEPH_WALL_TOP_RIGHT_3"]

    def __init__(
            self,
            screen: Screen,
            template_finder: TemplateFinder,
            pather: Pather,
            town_manager: TownManager,
            ui_manager: UiManager,
            char: IChar,
            pickit: PickIt
    ):
        self._config = Config()
        self._screen = screen
        self._template_finder = template_finder
        self._pather = pather
        self._town_manager = town_manager
        self._ui_manager = ui_manager
        self._char = char
        self._pickit = pickit
        self.used_tps = 0

        self._walls = [
            self.WALL_BOTTOM_LEFT,
            self.WALL_BOTTOM_RIGHT,
            self.WALL_TOP_LEFT,
            self.WALL_TOP_RIGHT
        ]

    def approach(self, start_loc: Location) -> Union[bool, Location, bool]:
        Logger.info("Run Mephisto")
        if not self._char.can_teleport():
            raise ValueError("Mephisto requires teleport")
        if not self._town_manager.open_wp(start_loc):
            return False
        wait(0.4)
        self._ui_manager.use_wp(3, 8)  # use Durance of Hate Level 2 (9th in A3)
        return Location.A3_MEPHISTO_START

    def battle(self, do_pre_buff: bool) -> Union[bool, tuple[Location, bool]]:
        if do_pre_buff:
            self._char.pre_buff()
        self._travel()

    def _travel(self):
        self._char.pre_move()
        current_orientation = Orientation.create_by_number(self._screen, 0, False)
        previous_orientation = current_orientation
        previous_jump_stuck = False
        while True:
            Logger.debug("Using orientation: " + str(current_orientation))
            score, img = self._tele(current_orientation.monitor_position)

            if self._find_stairs(img):
                break

            if score < 0.04:
                previous_orientation = current_orientation
                current_orientation = self._force_unstuck(current_orientation)
                previous_jump_stuck = True
                continue

            # Try to jump again into the previous direction
            if previous_jump_stuck:
                score, img = self._tele(previous_orientation.monitor_position)
                if score < 0.04:
                    previous_jump_stuck = False
                else:
                    current_orientation = self._orient_based_on_walls(img, current_orientation)
                    if current_orientation is None:
                        current_orientation = previous_orientation

            next_orientation = self._orient_based_on_walls(img, current_orientation)
            if next_orientation is not None:
                previous_orientation = current_orientation
                current_orientation = self._orient_based_on_walls(img, current_orientation)

    def _tele(self, position: Tuple[float, float]) -> Tuple[int, np.ndarray]:
        x_m, y_m = position[0], position[1]
        x_m += int(random.random() * 6 - 3)
        y_m += int(random.random() * 6 - 3)
        t0 = self._screen.grab()
        self._char.move((x_m, y_m))
        t1 = self._screen.grab()

        # check difference between the two frames to determine if tele was good or not
        diff = cv2.absdiff(t0, t1)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff, 13, 255, cv2.THRESH_BINARY)
        score = (float(np.sum(mask)) / mask.size) * (1 / 255.0)
        Logger.debug("Score: " + str(score))

        return score, t1

    def _find_stairs(self, img: np.ndarray) -> bool:
        # Try to find the stairs to Level 3
        stairs_found = self._template_finder.search(["MEPH_STAIRS_0", "MEPH_STAIRS_1", "MEPH_STAIRS_2"], img)
        if stairs_found.valid:
            Logger.debug("The Durance of Hate Level 3 stairs found!")
            self._char.move(self._screen.convert_screen_to_monitor(stairs_found.position))
            wait(0.4, 0.6)
            found_loading_screen_func = lambda: self._ui_manager.wait_for_loading_screen(2.0) or \
                                                self._template_finder.search_and_wait(
                                                    ["NI2_SEARCH_0", "NI2_SEARCH_1"], threshold=0.8, time_out=0.5,
                                                    use_grayscale=True).valid
            Logger.debug("Clicking stairs...")
            self._char.select_by_template(["MEPH_STAIRS_0", "MEPH_STAIRS_1", "MEPH_STAIRS_2"],
                                          found_loading_screen_func,
                                          threshold=0.63)
            return True

        return False

    def _orient_based_on_walls(self, img: np.ndarray, current_orientation: Orientation):
        # Let's crop 600x600px around the character and try to find walls on it
        screen = self._screen.convert_abs_to_screen((0, 0))
        cropped = img[screen[1] - 300:screen[1] + 300, screen[0] - 300:screen[0] + 300]

        # Check if the template we are following still exists, in that case we can continue following it
        for t in current_orientation.template_to_follow:
            if self._template_finder.search(t, cropped).valid:
                Logger.debug("Everything looks okay, next hop")
                return current_orientation

        Logger.debug("Current orientation texture lost! Trying to match a new one")
        match = None
        for w in self._walls:
            for w_img in w:
                new_match = self._template_finder.search(w_img, img, use_grayscale=True)
                if match is None or match.score < new_match.score:
                    match = new_match
        Logger.debug("Best Match: " + str(match))

        if not match.valid:
            Logger.debug("We are lost :( Let's continue until we found a new wall to follow")
            return None

        if match.name in self.WALL_BOTTOM_RIGHT:
            if current_orientation.reversed:
                return Orientation.create_by_number(self._screen, 3, True)
            else:
                return Orientation.create_by_number(self._screen, 1, False)

        if match.name in self.WALL_TOP_RIGHT:
            if current_orientation.reversed:
                return Orientation.create_by_number(self._screen, 2, True)
            else:
                return Orientation.create_by_number(self._screen, 0, False)

        if match.name in self.WALL_TOP_LEFT:
            if current_orientation.reversed:
                return Orientation.create_by_number(self._screen, 1, True)
            else:
                return Orientation.create_by_number(self._screen, 3, False)

        if match.name in self.WALL_BOTTOM_LEFT:
            if current_orientation.reversed:
                return Orientation.create_by_number(self._screen, 0, True)
            else:
                return Orientation.create_by_number(self._screen, 2, False)

        return current_orientation

    def _force_unstuck(self, current_orientation: Orientation):
        # Force orientation change. Best effort to detect logical direction
        Logger.debug("Player stuck! Changing orientation based on current orientation!")
        if current_orientation.number == 0:
            return Orientation.create_by_number(self._screen, 1, False)
        if current_orientation.number == 1:
            return Orientation.create_by_number(self._screen, 2, False)
        if current_orientation.number == 2:
            return Orientation.create_by_number(self._screen, 3, False)
        if current_orientation.number == 3:
            return Orientation.create_by_number(self._screen, 0, False)
