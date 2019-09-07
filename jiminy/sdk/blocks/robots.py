from jiminy.sdk.wrappers import Block
import pyautogui as pg
import pyperclip as pc
import time

class GoogleKeepRobot(Block):
    def __init__(self, theme="dark"):
        super(GoogleKeepRobot, self).__init__(name="GoogleKeepRobot",
                input_dict={
                    "title" : (1,),
                    "selected-text" : (1,),
                    })
        if theme == "dark":
            self.icon = "assets/keep-dark.png"
        else:
            self.icon = "assets/keep-light.png"
        self.theme = theme

    def _forward(self, inputs):
        typeText = "{}\n{}".format(inputs["title"], inputs["selected-text"])
        tabx, taby = None, None
        for icon in ['assets/keep-dark.png', 'assets/keep-dark-2.png', 'assets/keep-dark-3.png']:
            try:
                tabx, taby = pg.locateCenterOnScreen(icon)
                pg.click(tabx // 2, taby // 2)
                time.sleep(0.2)
                break
            except:
                continue
        if tabx is None:
            tabx, taby = pg.locateCenterOnScreen('assets/{}-new-tab.png'.format(self.theme))
            pg.click(tabx // 2, taby // 2)
            time.sleep(0.4)
            pg.typewrite("https://keep.google.com/u/0/#home\n")
            time.sleep(1)
        x, y = pg.locateCenterOnScreen('assets/keep-note-keyframe.png')
        cx, cy = x, y
        # except:
        #     tabx, taby = pg.locateCenterOnScreen('assets/{}-new-tab.png'.format(self.theme))
        #     pg.moveTo(tabx, taby), pg.typewrite("https://keep.google.com/u/0/#home\n")

        pg.moveTo(x // 2, y // 2, 0.5)
        pg.mouseDown() ; pg.mouseUp() # ; pg.mouseDown() ; pg.mouseUp() ;
        pc.copy(inputs["selected-text"])
        pg.hotkey("command", "v")
        pg.mouseDown() ; pg.mouseUp() # ; pg.mouseDown() ; pg.mouseUp()
        pc.copy(inputs["title"])
        pg.hotkey("command", "v")
        x, y = pg.locateCenterOnScreen('assets/keep-close.png')
        pg.moveTo(x // 2, y //2, 0.01)
        pg.mouseDown() ; pg.mouseUp() ; # pg.mouseDown() ; pg.mouseUp()
        pg.click((cx // 2) - 150, cy // 2)
        pg.mouseDown() ; pg.mouseUp() ; # pg.mouseDown() ; pg.mouseUp()


if __name__ == "__main__":
    robot = GoogleKeepRobot()
    t = time.time()
    robot.forward({
        "title" : "Paul Graham - Economic Inequality",
        "selected-text" : "Sometimes this is done for ideological reasons. Sometimes it's\nbecause the writer only has very high-level data and so draws\nconclusions from that, like the proverbial drunk who looks for his\nkeys under the lamppost, instead of where he dropped them,\nbecause the light is better there. Sometimes it's because the\nwriter doesn't understand critical aspects of inequality, like the\nrole of technology in wealth creation. Much of the time, perhaps\nmost of the time, writing about economic inequality combines all\nthree."
        })
    print(time.time() - t)
