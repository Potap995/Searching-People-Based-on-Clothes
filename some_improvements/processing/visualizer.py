import cv2
import numpy as np
from profilehooks import profile
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class Visualizer:

    def __init__(self):
        pass

    def draw_tracks(self, frame, bboxes, colors=None, border=1, relative=True):
        if colors is None:
            colors = np.random.randint(256, size=(len(bboxes), 3))
        size = (1, 1)
        if relative:
            size = (frame.shape[1], frame.shape[0])  # width, height
        for idx, bbox in enumerate(bboxes):
            bbox_id = int(bbox[0])
            colour = tuple(map(int, colors[idx]))

            cv2.rectangle(frame,
                          (int(bbox[1]*size[0]), int(bbox[2]*size[1])),
                          (int((bbox[1]+bbox[3])*size[0]), int((bbox[2]+bbox[4])*size[1])),
                          colour,
                          border+1)

            # cv2.putText(img, str(track[1]), (track[2], track[3]), 0, 5e-3 * 200, (0, 255, 0), 1)

        return frame


# class from detectron2 Visualizer
class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


def random_color():
    return tuple(np.random.rand(3))


# function from detectron2 GenericMask
def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]
    return res, has_holes


_SMALL_OBJECT_AREA_THRESH = 1000


def draw_on_img(img, masks, bboxes, labels, alpha):
    output = VisImage(img, scale=1)
    font_size = np.sqrt(output.height * output.width) / 50
    print(font_size)

    num_instances = len(masks)
    if len(masks) != len(bboxes) or len(masks) != len(labels):
        print(f"masks: {len(masks)}, bboxes: {len(bboxes)}, labels: {len(labels)}")
        raise Exception("Колво треков и информации о них не совпадает")

    assigned_colors = [random_color() for _ in range(num_instances)]

    areas = np.prod(bboxes[:, 2:] - bboxes[:, :2], axis=1)
    sorted_idxs = np.argsort(-areas).tolist()
    bboxes = bboxes[sorted_idxs]
    labels = [labels[k] for k in sorted_idxs]
    masks = [masks[idx] for idx in sorted_idxs]

    for idx in range(num_instances):
        cur_color = assigned_colors[idx]
        # bbox

        x0, y0, x1, y1 = bboxes[idx]
        width = x1 - x0
        height = y1 - y0

        output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=cur_color,
                linewidth=max(font_size / 5, 1),
                alpha=0.5,
                linestyle="-",
            )
        )

        # mask

        polygons, has_holes = mask_to_polygons(masks[idx])
        for segment in polygons:
            segment = segment.reshape(-1, 2)
            polygon = mpl.patches.Polygon(
                segment,
                fill=True,
                facecolor=mplc.to_rgb(cur_color) + (alpha,),
                edgecolor=mplc.to_rgb(cur_color) + (1,),
                linewidth=max(font_size / 5, 1),
            )
            output.ax.add_patch(polygon)

        # label

        text_pos = (x0, y0)
        horiz_align = "left"

        # instance_area = (y1 - y0) * (x1 - x0)
        # if (
        #         instance_area < _SMALL_OBJECT_AREA_THRESH
        #         or y1 - y0 < 40
        # ):
        #     if y1 >= output.height - 5:
        #         text_pos = (x1, y0)
        #     else:
        #         text_pos = (x0, y1)
        #
        # height_ratio = (y1 - y0) / np.sqrt(output.height * output.width)
        # font_size = (
        #         np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
        #         * 0.5
        #         * font_size
        # )

        color = np.maximum(list(mplc.to_rgb(cur_color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = text_pos
        output.ax.text(
            x,
            y,
            labels[idx],
            size=font_size*2,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horiz_align,
            color=color,
            zorder=10,
            rotation=0,
        )

    return output.get_image()
