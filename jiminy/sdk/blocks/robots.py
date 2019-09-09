from jiminy.sdk.wrappers import Block
import pyautogui as pg
import pyperclip as pc
import time
import numpy as np

animation = pg.easeInOutQuad
animationTime = 0.2

class GoogleKeepRobot(Block):
    def __init__(self, chrome, theme="dark", small=True, *args, **kwargs):
        super(GoogleKeepRobot, self).__init__(name="GoogleKeepRobot",
                input_dict={
                    "title" : (1,),
                    "selected-text" : (1,),
                    })
        self.theme = theme
        self.chrome = chrome
        self.small = small
        if small :
            self.theme += "/small"
            self.regionTabs = (self.chrome[0], self.chrome[1], (self.chrome[2] - self.chrome[0]), 75)
        else :
            self.regionTabs =(self.chrome[0]*2, self.chrome[1]*2, 2*(self.chrome[2] - self.chrome[0]), 150)

    def _forward(self, inputs):
        typeText = "{}\n{}".format(inputs["title"], inputs["selected-text"])
        px, py = None, None
        for path in ['current-tab', 'current-tab-other']:
            try:
                px, py = pg.locateCenterOnScreen('../blocks/assets/{}/{}.png'.format(self.theme, path),
                        region=self.regionTabs)
                break
            except:
                continue
        tabx, taby = None, None
        t = time.time()
        for icon in ['keep-1.png', 'keep-2.png', 'keep-3.png', 'keep-4.png']:
            try:
                tabx, taby = pg.locateCenterOnScreen('../blocks/assets/{}/{}'.format(self.theme, icon),
                        region=self.regionTabs)
                pg.moveTo(tabx // (1 if self.small else 2), taby // (1 if self.small else 2), animationTime, animation)
                self.hard_click()
                break
            except:
                continue
        print("Icon finding: ", time.time() - t)
        if tabx is None:
            for newT in ['new-tab', 'new-tab-light']:
                try:
                    tabx, taby = pg.locateCenterOnScreen('../blocks/assets/{}/{}.png'.format(self.theme, newT), region=self.regionTabs)
                    break
                except:
                    continue
            pg.moveTo(tabx // (1 if self.small else 2), taby // (1 if self.small else 2), animationTime, animation)
            self.hard_click()
            pg.write("https://keep.google.com/u/0/#home\n")
            time.sleep(15)
        t = time.time()
        cross = pg.screenshot(region=(self.chrome[0]*(1 if self.small else 2) + 50, (self.chrome[1] + 85)*(1 if self.small else 2), 15*(1 if self.small else 2), 15*(1 if self.small else 2))).convert('RGB')
        if (np.array(cross) > 200).all():
            self.keep_theme = "light"
        else:
            self.keep_theme = "dark"
        print("Using {} theme".format(self.keep_theme))
        pg.press('esc')
        pg.hotkey('command', 'up')
        time.sleep(0.5)
        x, y = pg.locateCenterOnScreen('../blocks/assets/{}/keep-note-keyframe.png'.format(self.keep_theme),
                region=(self.chrome[0]*(1 if self.small else 2), self.chrome[1]*(1 if self.small else 2), (1 if self.small else 2)*self.chrome[2], (1 if self.small else 2)*self.chrome[3]))
        cx, cy = x, y

        pg.moveTo(x // (1 if self.small else 2), y // (1 if self.small else 2), animationTime, animation)
        self.hard_click()
        pc.copy(inputs["selected-text"])
        pg.hotkey("command", "v")
        self.hard_click()
        pc.copy(inputs["title"])
        pg.hotkey("command", "v")
        t = time.time()
        x, y = pg.locateCenterOnScreen('../blocks/assets/{}/keep-close.png'.format(self.keep_theme),
                region=(self.chrome[0]*2, self.chrome[1]*2, 2*self.chrome[2], 2*self.chrome[3]))
        pg.moveTo(x // (1 if self.small else 2), y //(1 if self.small else 2), animationTime, animation)
        self.hard_click()
        pg.moveTo((px // (1 if self.small else 2)) - 15, py // (1 if self.small else 2), animationTime, animation)
        self.hard_click()

    def hard_click(self):
        pg.mouseDown() ; pg.mouseUp()


if __name__ == "__main__":
    robot = GoogleKeepRobot((0, 0, 200, 200))
    robot.forward({
        "title" : "Paul Graham - Economic Inequality",
        "selected-text" : "Sometimes this is done for ideological reasons. Sometimes it's\nbecause the writer only has very high-level data and so draws\nconclusions from that, like the proverbial drunk who looks for his\nkeys under the lamppost, instead of where he dropped them,\nbecause the light is better there. Sometimes it's because the\nwriter doesn't understand critical aspects of inequality, like the\nrole of technology in wealth creation. Much of the time, perhaps\nmost of the time, writing about economic inequality combines all\nthree."
        })
