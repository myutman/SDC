from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2


def get_duck_size(im):
    im1 = im[:, :, :2].mean(axis=-1).astype(np.uint8)
    im1 = cv2.dilate(im1, np.ones((25, 25)))
    im1 = cv2.erode(im1, np.ones((25, 25)))
    _, im_no_background = cv2.threshold(im1, 100, 255, cv2.THRESH_BINARY)
    n_comp, components, meta, coord = cv2.connectedComponentsWithStats(im_no_background)

    size = 0
    for i in range(1, n_comp):
        if not (np.any(components[0, :] == i) or np.any(components[:, 0] == i)):
            size = max(size, np.sum(components == i) / np.size(components))
    return size


class DontCrushDuckieTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task['env']
        # getting the initial picture
        img, _, _, _ = env.step([0,0])

        condition = True
        while condition:
            img, reward, done, info = env.step([1, 0])
            # img in RGB
            # add here some image processing

            duck_size = get_duck_size(img)
            condition = (duck_size < 0.07)
            env.render()

        env.step([0, 0])
        env.render()

