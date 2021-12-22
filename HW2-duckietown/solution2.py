from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2
import logging
import collections


LOGGER = logging.getLogger()


def get_duck_size(im):
    im1 = im[:, :, :2].mean(axis=-1).astype(np.uint8)
    im1 = cv2.dilate(im1, np.ones((25, 25)))
    im1 = cv2.erode(im1, np.ones((25, 25)))
    _, im_no_background = cv2.threshold(im1, 90, 255, cv2.THRESH_BINARY)
    n_comp, components, meta, coord = cv2.connectedComponentsWithStats(im_no_background)

    size = 0
    for i in range(1, n_comp):
        if not (np.any(components[0, :] == i) or np.any(components[:, 0] == i)):
            size = max(size, np.sum(components == i) / np.size(components))
    #LOGGER.warn(f"size={size}")
    return size


def get_matrix_by_points(points):
    #LOGGER.warn(f"points={points}")
    X = np.concatenate([points[:, 0].reshape(-1, 1), np.ones_like(points[:, 0]).reshape(-1, 1)], axis=1)
    b = points[:, 1].reshape(-1, 1)
    if points[0, 0] == points[1, 0]:
        A = None
        mse = np.inf
    else:
        A = (b.T @ X @ np.matrix(X.T @ X).I).A
        mse = np.sqrt(((A @ X.T - b.T) ** 2).mean())

    X1 = np.concatenate([points[:, 1].reshape(-1, 1), np.ones_like(points[:, 1]).reshape(-1, 1)], axis=1)
    b1 = points[:, 0].reshape(-1, 1)
    if points[0, 1] == points[1, 1]:
        A1 = None
        mse1 = np.inf
    else:
        A1 = (b1.T @ X1 @ np.matrix(X1.T @ X1).I).A
        mse1 = np.sqrt(((A1 @ X1.T - b1.T) ** 2).sum())

    if mse < mse1:
        return mse, [A[0][0], -1, A[0][1]]
    else:
        return mse1, [-1, A1[0][0], A1[0][1]]


def get_sky(im):
    im3 = im[:, :, 2]
    _, im3 = cv2.threshold(im3, 220, 255, cv2.THRESH_BINARY)
    im3 = cv2.dilate(im3, np.ones((20, 20)))
    im3 = cv2.erode(im3, np.ones((22, 22)))
    return im3


def get_sky_borders_mask(im):
    im1 = im[:, :, 2].astype(np.uint8)
    im1[get_sky(im) == 255] = 255
    im1 = cv2.dilate(im1, np.ones((15, 15)))
    im1 = cv2.erode(im1, np.ones((15, 15)))
    _, im1 = cv2.threshold(im1, 140, 255, cv2.THRESH_BINARY)
    n_comp, components, meta, coord = cv2.connectedComponentsWithStats(im1)

    sky_comp = -1
    for i in range(1, n_comp):
        if (np.any(components[0, :] == i)):
            sky_comp = i

    border_comp = -1
    for i in range(1, n_comp):
        if (np.any(components[:, 0] == i) and np.any(components[:, -1] == i)) and i != sky_comp:
            border_comp = i


    if border_comp != -1:
        i = border_comp
    else:
        i = sky_comp

    im1[:, :] = 0

    height, width = im1.shape
    for y in range(width):
        maxx = (np.arange(height) * (components[:, y] == i)).max()
        im1[:maxx + 1, y] = 255

    return im1


def check_comp(components, i):
    checks = np.zeros_like(components, dtype=int)

    comp = np.zeros_like(components, dtype=np.uint8)
    comp[components == i] = 255
    comp = cv2.erode(comp, np.ones((3, 3)))

    height, width = components.shape
    for x in np.arange(height)[np.any(comp == 255, axis = -1)]:
        r = np.arange(width)
        r[comp[x, :] != 255] = 0
        y = np.max(r)
        checks[x, :y + 1] += 1

        r[comp[x, :] != 255] = width + 1
        y = np.min(r)
        checks[x, y:] += 1

    for y in np.arange(width)[np.any(comp, axis = 0)]:
        r = np.arange(height)
        r[comp[:, y] != 255] = 0
        x = np.max(r)
        checks[:x + 1, y] += 1

        r[comp[:, y] != 255] = height + 1
        x = np.min(r)
        checks[x:, y] += 1

    return not np.any((checks == 4) & (comp != 255))


def get_brick_borders(im):
    im3 = get_sky_borders_mask(im)

    im1 = im.mean(axis=-1).astype(np.uint8)
    im1[im3 == 255] = 0

    _, im1 = cv2.threshold(im1, 100, 255, cv2.THRESH_BINARY)

    n_comp, components, meta, coord = cv2.connectedComponentsWithStats(im1)

    midpoints = []

    mx = 0
    mxi = -1
    for i in range(1, n_comp):
        size = np.sum(components == i) / components.size
        if size > 0.0005 and (not np.any(components[:, 0] == i)) and (not np.any(components[0, :] == i)) and (not np.any(components[:, -1] == i)) and (not np.any(components[-1, :] == i)) and check_comp(components, i):
            if size > mx:
                mx = size
                mxi = i

    if mxi == -1:
        return False, None, None, None, None

    kek = np.zeros_like(im1, dtype=np.uint8)

    height, width = components.shape
    get_xy = lambda paired: [paired // width, paired % width]

    rightmost = np.arange(width).reshape(1, -1).repeat(height, axis=0)
    rightmost[components != mxi] = -1
    right_x, right_y = get_xy(rightmost.argmax())

    vertical = np.abs(np.arange(height, dtype=np.float32).reshape(-1, 1) - right_x) / np.abs(np.arange(width, dtype=np.float32).reshape(1, -1) - (right_y + 10))
    vertical[components != mxi] = -1
    vertical[right_x, right_y] = -1
    right_x1, right_y1 = get_xy(vertical.argmax())


    leftmost = np.arange(width).reshape(1, -1).repeat(height, axis=0)
    leftmost[components != mxi] = width + 1
    left_x, left_y = get_xy(leftmost.argmin())

    vertical = np.abs(np.arange(height, dtype=np.float32).reshape(-1, 1) - left_x) / np.abs(np.arange(width, dtype=np.float32).reshape(1, -1) - (left_y - 10))
    vertical[components != mxi] = -1
    vertical[left_x, left_y] = -1
    left_x1, left_y1 = get_xy(vertical.argmax())



    upmost = np.arange(height).reshape(-1, 1).repeat(width, axis=-1)
    upmost[components != mxi] = height + 1
    up_x, up_y = get_xy(upmost.argmin())

    horisontal = np.abs(np.arange(width, dtype=np.float32).reshape(1, -1) - up_y) / np.abs(np.arange(height, dtype=np.float32).reshape(-1, 1) - (up_x - 10))
    horisontal[components != mxi] = -1
    horisontal[up_x, up_y] = -1
    up_x1, up_y1 = get_xy(horisontal.argmax())

    downmost = np.arange(height).reshape(-1, 1).repeat(width, axis=-1)
    downmost[components != mxi] = -1
    down_x, down_y = get_xy(downmost.argmax())

    horisontal = np.abs(np.arange(width, dtype=np.float32).reshape(1, -1) - down_y) / np.abs(np.arange(height, dtype=np.float32).reshape(-1, 1) - (down_x + 10))
    horisontal[components != mxi] = -1
    horisontal[down_x, down_y] = -1
    down_x1, down_y1 = get_xy(horisontal.argmax())

    _, A_left = get_matrix_by_points(np.array([[left_x, left_y], [left_x1, left_y1]], dtype=np.float32))
    _, A_right = get_matrix_by_points(np.array([[right_x, right_y], [right_x1, right_y1]], dtype=np.float32))
    _, A_up = get_matrix_by_points(np.array([[up_x, up_y], [up_x1, up_y1]], dtype=np.float32))
    _, A_down = get_matrix_by_points(np.array([[down_x, down_y], [down_x1, down_y1]], dtype=np.float32))

    return True, A_left, A_right, A_up, A_down


def line_inter(A1, A2):
    x = (A1[2] * A2[0] - A2[2] * A1[0]) / (A1[0] * A2[1] - A2[0] * A1[1])
    y = (A1[2] * A2[1] - A2[2] * A1[1]) / (A2[0] * A1[1] - A1[0] * A2[1])
    return x, y


def get_translation_matrix(im, brick_size=30):
    src_im = im.copy()

    is_visible, A_left, A_right, A_top, A_bottom = get_brick_borders(im)

    if not is_visible:
        return False, None

    #LOGGER.warn(f"matrices={[A_left, A_right, A_top, A_bottom]}")

    x1, y1 = line_inter(A_left, A_bottom)
    x2, y2 = line_inter(A_right, A_bottom)
    x3, y3 = line_inter(A_left, A_top)
    x4, y4 = line_inter(A_right, A_top)

    src = np.array([
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4]
    ], dtype=np.float32)

    height, width, _ = im.shape

    dst = np.array([
        [-0.0625, 0],
        [0.0625, 0],
        [-0.0625, 0.125],
        [0.0625, 0.125]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    #LOGGER.warn(f"mat={M}")

    inp = np.array([height - 1, width // 2, 1], dtype=np.float32).reshape(-1, 1)
    #LOGGER.warn(f"inp={inp}")

    out = (M @ inp).reshape(-1)
    #LOGGER.warn(f"out={out}")

    return True, [(out[0] / out[2]), (out[1] / out[2])]

class DontCrushDuckieTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task['env']
        # getting the initial picture
        img, _, _, _ = env.step([0,0])

        condition = True

        dq = collections.deque(maxlen=1)
        speed = 0.2

        while condition:
            img, reward, done, info = env.step([1, 0])
            # img in RGB
            # add here some image processing
            x, _ = get_translation_matrix(img)
            if not np.isnan(x):
                dq.append(x)
            if len(dq) > 0:
                x_cum = np.mean(dq)
                LOGGER.warn(f"x={x_cum}")

            duck_size = get_duck_size(img)
            condition = (duck_size < 0.02)
            env.render()

        condition = True

        Kp = 0.1 / speed
        Kd = 3.0
        Ki = 0.01
        cnt_threshold = 30

        target = -0.2
        ei = 0.
        old_e = target - x_cum

        cnt = 0

        while condition:
            is_visible, out = get_translation_matrix(img)

            if not is_visible:
                img, reward, done, info = env.step([speed / 4, -0.04 / speed])
                continue

            x, _ = out


            if not np.isnan(x):
                dq.append(x)
            if len(dq) > 0:
                x_cum = np.mean(dq)

            if x_cum < target:
                cnt += 1
            else:
                cnt = 0

            condition = cnt < cnt_threshold
            if not condition:
                break

            e = target - x_cum
            LOGGER.warn(f"x={x_cum}, e={e}, diff={e - old_e}, ei={ei}")
            U = Kp * (e + Kd * (e - old_e) + Ki * ei)
            ei += e
            old_e = e

            img, reward, done, info = env.step([speed, -1 * U])

            # img in RGB
            # add here some image processing

            env.render()

        is_visible, out = get_translation_matrix(img)
        while not is_visible:
            img, reward, done, info = env.step([speed / 4, -0.04 / speed])

        speed = 0.45
        Kp = 0.1 / speed
        target = 0.4
        cnt_threshold = 10

        for _ in range(4):
            ei = 0.
            e_old = target - x_cum
            condition = True

            cnt = 0
            while condition:
                is_visible, out = get_translation_matrix(img)

                if not is_visible:
                    img, reward, done, info = env.step([speed / 4, 0.04 / speed])
                    continue
                x, _ = out


                if not np.isnan(x):
                    dq.append(x)
                if len(dq) > 0:
                    x_cum = np.mean(dq)

                if x_cum > target:
                    cnt += 1
                else:
                    cnt = 0

                condition = cnt < cnt_threshold
                if not condition:
                    break

                e = target - x_cum
                LOGGER.warn(f"x={x_cum}, e={e}, diff={e - old_e}, ei={ei}")
                U = Kp * (e + Kd * (e - old_e) + Ki * ei)
                ei += e
                old_e = e

                img, reward, done, info = env.step([speed, -1 * U])

                # img in RGB
                # add here some image processing

                env.render()


        
