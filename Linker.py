import numpy as np 
from scipy import signal

class linkedEdges:
    def __init__(self, magnitude, orientation):
        self.magnitude = magnitude
        self.orientation = orientation
    
    def calc_local_max(self, neighbor_x, neighbor_y):

        if len(neighbor_x.shape) == 2 or len(neighbor_x.shape) == 2:
            dimensions = 2
            local_height = neighbor_x.shape[0]
            local_width = neighbor_x.shape[1]
            # flatten since it's 2 dimensional 
            neighbor_x = neighbor_x.flatten()
            neighbor_y = neighbor_y.flatten()

        # take height and width of magnitude
        height, width = self.magnitude.shape

        # check if flattened regions are of different shapes
        if neighbor_x.shape != neighbor_y.shape:
            raise Exception('''The shapes of the local regions used to 
            calculated local maximum are not of equal size.''')
        
        # create floor and ceil thresholds, cast as int
        x_floor = np.floor(neighbor_x).astype(np.int32)
        x_ceil = np.ceil(neighbor_x).astype(np.int32)
        
        y_floor = np.floor(neighbor_y).astype(np.int32)
        y_ceil = np.ceil(neighbor_y).astype(np.int32)

        # remove negatives
        x_floor = np.where(x_floor < 0, 0, x_floor)
        x_ceil = np.where(x_ceil < 0, 0, x_ceil)
        y_floor = np.where(y_floor < 0, 0, y_floor)
        y_ceil = np.where(y_ceil < 0, 0, y_ceil)

        # reset to match total max and mins on the deriv image Mag
        x_floor = np.where(x_floor >= width - 1, width - 1, x_floor)
        x_ceil = np.where(x_ceil >= width - 1, width - 1, x_ceil)
        y_floor = np.where(y_floor >= height - 1, height - 1, y_floor)
        y_ceil = np.where(y_ceil >= height - 1, height - 1, y_ceil)

        cord_1 = self.magnitude[y_floor, x_floor]
        cord_2 = self.magnitude[y_floor, x_ceil]
        cord_3 = self.magnitude[y_ceil, x_floor]
        cord_4 = self.magnitude[y_ceil, x_ceil]

        lh = neighbor_y - y_floor
        lw = neighbor_x - x_floor
        hh = 1 - lh
        hw = 1 - lw

        # get the windows
        w1 = hh * hw
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw

        local_max = (cord_1 * w1) + (cord_2 * w2) + (cord_3 * w3) + (cord_4 * w4)

        if dimensions == 2:
            return local_max.reshape(local_height, local_width)
        return local_max
    
    def non_max_suppression(self):

        rows, cols = self.magnitude.shape
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        neighbor1_x = np.clip(x + np.cos(self.orientation), 0, cols - 1)
        neighbor1_y = np.clip(y + np.sin(self.orientation), 0, rows - 1)
        neighbor1 = self.calc_local_max(neighbor1_x, neighbor1_y)

        # find the opposite orientation
        opposite_orientation = np.add(self.orientation, np.pi)

        neighbor2_x = np.clip(x + np.cos(opposite_orientation), 0, cols - 1)
        neighbor2_y = np.clip(y + np.sin(opposite_orientation), 0, rows - 1)
        neighbor2 = self.calc_local_max(neighbor2_x, neighbor2_y)

        nms = np.empty([rows, cols], dtype=bool)

        for row in range(rows):
            for col in range(cols):
                if (self.magnitude[row, col] < neighbor1[row, col]) or (
                    self.magnitude[row, col] < neighbor2[row, col]
                ):
                    nms[row, col] = False
                else:
                    nms[row, col] = True

        return nms

    def edge_link(self, low_thresh, high_thresh):
        # suppress the magnitude where nms(i, j) == False 
        nms_sup = np.where(self.non_max_suppression(), self.magnitude, 0)
        
        # areas in between low and high thresh is the uncertain region
        uncertain = np.logical_and(low_thresh < nms_sup, nms_sup < high_thresh)
        
        # collect the edges we are certain about
        strong_edges = nms_sup >= high_thresh
        strong_edge_ori = self.orientation + np.pi / 2

        # neighbors pointing to the edge dir
        rows, cols = uncertain.shape
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        neighbor1_x = np.clip(x + np.cos(strong_edge_ori), 0, cols - 1)
        neighbor1_y = np.clip(y + np.sin(strong_edge_ori), 0, rows - 1)

        neighbor2_x = np.clip(x - np.cos(strong_edge_ori), 0, cols - 1)
        neighbor2_y = np.clip(y - np.sin(strong_edge_ori), 0, rows - 1)
        
        done = False

        # loop until uncertain points don't change anymore
        while not done:
            # calc max of the neighbor
            neighbor_1 = self.calc_local_max(neighbor1_x, neighbor1_y)
            neighbor_2 = self.calc_local_max(neighbor2_x, neighbor2_y)

            # if neigbor is aabove high and uncertain, we can add it. 
            is_strong_neighbor = np.logical_or(neighbor_1 >= high_thresh, neighbor_2 >= high_thresh)
            update = np.logical_and(uncertain, is_strong_neighbor)
            nms_sup = np.where(
                update, np.maximum(neighbor_1, neighbor_2), nms_sup
            )
            # update the neighbor so it is no longer an uncertain edge
            uncertain[update] = False
            # update the strong edges map and link this new certain neighbor
            strong_edges[update] = True
            if not np.any(update):
                done = True

        return strong_edges