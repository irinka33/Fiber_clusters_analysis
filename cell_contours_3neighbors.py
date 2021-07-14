import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import uuid
from scipy.spatial import distance as dist
from collections import OrderedDict
import cv2
import csv
import math
import xml.etree.ElementTree as ET
from datetime import date

# INPUT
# select the mode: MIX_ON = True for mix double typed fibrils with two corresponding main types
#                  MIX_ON = False  for analysing double typed fibrils as independent typed
MIX_ON = False
USE_UNKNOWN = False
cell_name = 'SS27'
scale = 1.25

dir_name = 'data\\row data\\' + cell_name
file_name = dir_name + '\\input' + cell_name + '.csv'
# image_name = 'data\input\imageSS01.jpg'
contours_file = dir_name + '\\contours' + cell_name + '.cz'

# OUTPUT
if MIX_ON: all = '_all_'
else: all = '_'
clusters_dir = 'data\\output\\csv\\' + str(date.today()) + '_5neighbors\\'
clusters_filename = clusters_dir + 'clusters' + cell_name + all + str(date.today()) +'_5neighbors.csv'

WARNINGS = ''

R_1 = 39.5
R_2A = 38
R_2X = 35
MAX_NEIGHBORS = 10
MIN_POINTS_PER_CLUSTER = 5
MIN_NEIGHBORS = 2
MIN_DISTANCE_BETWEEN_CONTOURS = 10 #12.5/scale # 12.5 um in pixels

CLUSTER_COUNT = 0

R_koeff_adj = 1  # adjusting coefficient for radius -
# should be calculated as weighted combination of minEnclosedCircle's radius and radius from 'area'


# Input csv file format:
#  'ElementID', 'Area', 'CenterX', 'CenterY', 'type',
# Fields leaving/adding for processing:
#  'ElementID', 'Area', 'CenterX', 'CenterY', 'type', 'contour_id', 'cluster_id'
#       0          1         2         3         4          5           6


def open_contours(contours_file):
    contours = []

    tree = ET.parse(contours_file)
    root = tree.getroot()

    # TODO - читать scale из cz файла

    for cntr in root.iter('Bezier'):
        id = cntr.attrib['Id']
        points = cntr.find('Geometry').find('Points').text.split(' ')

        contour = []
        for point in points:
            pnt = [round(float(point.split(',')[0])*scale), round(float(point.split(',')[1])*scale)]
            contour.append([pnt])

        contours.append(np.array(contour))
        #print(id)
    return contours


def plot_contours(contours):
    plt.figure(200)
    for contour in contours:
        cx = []
        cy = []
        for cnt in contour:
            cx.append(cnt[0][0])
            cy.append(cnt[0][1])

        plt.plot(cx, cy, marker='o', linestyle='-', markersize=2)
    plt.show()


def open_data(file_name):
    pandasData = pd.read_csv(file_name,
                       names=['ElementID', 'Area',
                              'CenterX', 'CenterY', 'type'], skiprows=1)

    # TODO - решить, где делать массшатбирование - здесь или в контурах
    #pandasData['CenterX'] /= scale #0.31 #1.25 #pandasData['CenterX'].real
    #pandasData['CenterY'] /= scale # 0.31 #1.25
    pandasData['type'] = pandasData['type'].str.strip()

    return pandasData


def plot_pandasData(pandasData, ax):
    D1 = R_1 * 2
    D2 = R_2A * 2
    D3 = R_2X * 2
    colors = {'type 1': 'y', 'type 2A': 'r', 'type 2X': 'fuchsia',
              'type 1+type 2A': 'b', 'type 2A+type 2X': 'purple', 'type 1+type 2X': 'orange',
              'Null': 'black'}

    size = {'type 1': D1, 'type 2A': D2, 'type 2X': D3,
            'type 1+type 2A': int((D1 + D2) / 2), 'type 2A+type 2X': int((D2 + D3) / 2),
            'type 1+type 2X': int((D1 + D3) / 2),
            'Null': int((D1 + D2 + D3) / 3)}

    pandasData_grouped = pandasData.groupby('type')

    for key, group in pandasData_grouped:
        group.plot(x='CenterX', y='CenterY', ax=ax, kind='scatter', subplots=True,
                   title='input data', label=key, c=colors[key], s=size[key] * R_koeff_adj, alpha=0.65, legend=True)


def connect_points_with_contours(pandasData, contours):

    cx = np.array([pandasData['CenterX'].to_numpy()])
    cy = np.array([pandasData['CenterY'].to_numpy()])
    points_centers = np.append(cx, cy, axis = 0).transpose()

    contours_centers = []
    for contour in contours:
        M = cv2.moments(contour)

        if M['m00'] == 0:
            cx = contour[0][0][0]
            cy = contour[0][0][1]
            print('There is a contour with less than 3 points, ', contour.tolist())
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        contours_centers.append([cx, cy])
    contours_centers = np.array(contours_centers)

    dist_matrix = dist.cdist(points_centers, contours_centers)

    rows = dist_matrix.min(axis=1).argsort()
    cols = dist_matrix.argmin(axis=1)[rows]

    usedRows = set()
    usedCols = set()

    contours_id = -1*np.ones(rows.shape[0])

    for (row, col) in zip(rows, cols):
        if row in usedRows or col in usedCols:
            continue
        contours_id[row] = int(col)

        usedRows.add(row)
        usedCols.add(col)

    pandasData['Contour_id'] = contours_id

    unusedRows = set(range(0, dist_matrix.shape[0])).difference(usedRows)
    unusedCols = set(range(0, dist_matrix.shape[1])).difference(usedCols)

    if unusedRows or unusedCols:
        print('Data and contours do not match:')
        print('     for points ', unusedRows)
        print('     for contours ', unusedCols)

    return pandasData


def grouped_data_by_type(pandasData):

    rowData_grouped = pandasData.groupby(['type'])

    type1_points = []
    type2A_points = []
    type2X_points = []

    type1type2A_points = []
    type2Atype2X_points = []
    type1type2X_points = []
    unknown_type_points = []

    for type, group in rowData_grouped:
        # print(group.head())
        type_gr = group.to_numpy()
        if type == 'type 1':
            if len(type1_points) == 0: type1_points = type_gr
            else: type1_points = np.append(type1_points, type_gr, axis=0)
        elif type == 'type 2A':
            if len(type2A_points) == 0: type2A_points = type_gr
            else: type2A_points = np.append(type2A_points, type_gr, axis=0)
        elif type == 'type 2X':
            if len(type2X_points) == 0: type2X_points = type_gr
            else: type2X_points = np.append(type2X_points, type_gr, axis=0)

        #---
        elif type == 'type 1+type 2A':
            if MIX_ON:
                if len(type1_points) == 0: type1_points = type_gr
                else: type1_points = np.append(type1_points, type_gr, axis=0)
                if len(type2A_points) == 0: type2A_points = type_gr
                else: type2A_points = np.append(type2A_points, type_gr, axis=0)
            else:
                if len(type1type2A_points) == 0: type1type2A_points = type_gr
                else: type1type2A_points = np.append(type1type2A_points, type_gr, axis=0)
        elif type == 'type 2A+type 2X':
            if MIX_ON:
                if len(type2A_points) == 0: type2A_points = type_gr
                else: type2A_points = np.append(type2A_points, type_gr, axis=0)
                if len(type2X_points) == 0: type2X_points = type_gr
                else: type2X_points = np.append(type2X_points, type_gr, axis=0)
            else:
                if len(type2Atype2X_points) == 0: type2Atype2X_points = type_gr
                else: type2Atype2X_points = np.append(type2Atype2X_points, type_gr, axis=0)
        elif type == 'type 1+type 2X':
            if MIX_ON:
                if len(type1_points) == 0: type1_points = type_gr
                else: type1_points = np.append(type1_points, type_gr, axis=0)
                if len(type2X_points) == 0: type2X_points = type_gr
                else: type2X_points = np.append(type2X_points, type_gr, axis=0)
            else:
                if len(type1type2X_points) == 0: type1type2X_points = type_gr
                else: type1type2X_points = np.append(type1type2X_points, type_gr, axis=0)
        else: # no type (type == Null)
            if MIX_ON:
                if not USE_UNKNOWN:
                    if len(unknown_type_points) == 0: unknown_type_points = type_gr
                    else: unknown_type_points = np.append(unknown_type_points, type_gr, axis=0)
                else:
                    if len(type1_points) == 0: type1_points = type_gr
                    else: type1_points = np.append(type1_points, type_gr, axis=0)
                    if len(type2A_points) == 0: type2A_points = type_gr
                    else: type2A_points = np.append(type2A_points, type_gr, axis=0)
                    if len(type2X_points) == 0: type2X_points = type_gr
                    else: type2X_points = np.append(type2X_points, type_gr, axis=0)
            else:
                if len(unknown_type_points) == 0: unknown_type_points = type_gr
                else: unknown_type_points = np.append(unknown_type_points, type_gr, axis=0)

        #---
    if MIX_ON:
        return (type1_points, 'type 1'), (type2A_points, 'type 2A'), (type2X_points, 'type 2X'), \
               (unknown_type_points, 'Null')
    else:
        return (type1_points, 'type 1'), (type2A_points, 'type 2A'), (type2X_points, 'type 2X'), \
               (type1type2A_points, 'type 1+type 2A'), (type1type2X_points, 'type 1+type 2X'), \
               (type2Atype2X_points, 'type 2A+type 2X'),\
               (unknown_type_points, 'Null')


def plot_listData(data_grouped, contours, ax):
    colors = {'type 1': 'y', 'type 2A': 'r', 'type 2X': 'fuchsia',
              'type 1+type 2A': 'b', 'type 2A+type 2X': 'purple', 'type 1+type 2X': 'orange',
              'Null': 'black'}

    # fig, ax = plt.subplots()
    for (group, type) in data_grouped:
        for point in group:
            x = point[2]
            y = point[3]
            ax.plot(x, y, color=colors[type], alpha=0.45, marker='s', markersize=7)

            if point[5] == -1:
                continue
            contour = contours[int(point[5])]
            cx = []
            cy = []
            for cnt in contour:
                cx.append(cnt[0][0])
                cy.append(cnt[0][1])

            ax.plot(cx, cy, color=colors[type], alpha=0.45, marker='o',linestyle='-', markersize=2)
            #ax.legend()

    #return ax


def plot_clusters(clusters, ax):
    D1 = R_1*2#20
    D2 = R_2A*2#25
    D3 = R_2X*2#30

    # fig, ax = plt.subplots()
    colors = {'type 1': 'y', 'type 2A': 'r', 'type 2X': 'fuchsia',
              'type 1+type 2A': 'b', 'type 2A+type 2X': 'purple', 'type 1+type 2X': 'orange',
              'Null': 'black'}

    size = {'type 1': D1, 'type 2A': D2, 'type 2X': D3}

    for (cluster_group, type, data) in clusters:
        for key in cluster_group:
            x =[]
            y =[]
            (cnt, area, members) = cluster_group[key]
            for idx in members:
                cx = data[idx][2]
                cy = data[idx][3]
                x.append(cx)
                y.append(cy)
                ax.text(cx-3, cy-3, key)
            #ax.scatter(x, y, color=colors[type], alpha=0.65, marker = 's')
            ax.plot(x, y, color=colors[type], alpha=0.45, linestyle='-', marker='X', markersize=15)
            #ax.scatter(x, y, color=colors[type], alpha=0.45, marker='X')


def plot_clusters1(clusters, image):
    cv2.namedWindow("Clusters", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    cv2.resizeWindow("Clusters", 900, 600)

    D1 = R_1*2#20
    D2 = R_2A*2#25
    D3 = R_2X*2#30

    # fig, ax = plt.subplots()
    colors = {'type 1': (0,255,0), 'type 2A': (0,0,255), 'type 2X': (255,0,255)}

    size = {'type 1': D1, 'type 2A': D2, 'type 2X': D3}

    for (cluster_group, type, data) in clusters:
        for key in cluster_group:

            (cnt, area, members) = cluster_group[key]
            for idx in members:
                x = int(data[idx][2])
                y = int(data[idx][3])
                center = (int(x), int(y))
                radius = 20
                cv2.circle(image, center, radius, colors[type], 2)
            #ax.scatter(x, y, color=colors[type], alpha=0.65, marker = 's')
            #ax.plot(x, y, color=colors[type], alpha=0.45, linestyle='-', marker='s', markersize=5)
    cv2.imshow("Clusters", image)
    cv2.waitKey(0)


def check_neighbors(type_data, neighbors_idxs,  neighbors_dists, contours):

    points_neighbors = OrderedDict()

    radius = {'type 1': R_1, 'type 2A': R_2A, 'type 2X': R_2X,
              'type 1+type 2A': int((R_1+R_2A)/2), 'type 2A+type 2X': int((R_2A+R_2X)/2),
              'type 1+type 2X': int((R_1+R_2X)/2),
              'Null': int((R_1+R_2A+R_2X)/3)}

    # for each point check all their neighbors if they touch the point
    dist_matrix = np.ones((type_data.shape[0], type_data.shape[0]))*(-1)

    num_points = type_data.shape[0]
    num_neighbors = neighbors_idxs.shape[1]
    for i in range(0, num_points):
        point = type_data[i]
        type = point[4]
        cnt = 0
        neighbors = []
        contour_id = int(point[5])
        # TODO: if there is no contours for current point - we can use distance between centers???
        # it skips this point for now
        if contour_id == -1:
            continue
        point_contour = contours[contour_id]
        for j in range(0, num_neighbors):
            # neighbor = type_data[neighbors_idxs[i][j]] - we will need it if will consider to use 'Area' for radius calculating
            #distance = neighbors_dists[i][j]
            #radiuss = math.sqrt(type_data[neighbors_idxs[i][j]][1]/math.pi) #'area'
            #if 2*radius[type] >= distance:
            #if 2 * radiuss * R_koeff_adj >= distance:

            contour_id = int(type_data[neighbors_idxs[i][j]][5])
            # TODO: if there is no contours for neighbor point - we can use distance between centers???
            # it skips this neighbor for now
            if contour_id == -1:
                continue
            neighbor_contour = contours[contour_id]

            if dist_matrix[i][neighbors_idxs[i][j]] == -1:
                min_dist = min_dist_btwn_contours(point_contour, neighbor_contour)
                dist_matrix[i][neighbors_idxs[i][j]] = min_dist
                dist_matrix[neighbors_idxs[i][j]][i] = min_dist
            else:
                min_dist = dist_matrix[i][neighbors_idxs[i][j]]

            if min_dist <= MIN_DISTANCE_BETWEEN_CONTOURS:
                neighbors.append(neighbors_idxs[i][j])
                cnt += 1
            #else:
                #break
        points_neighbors[i] = (cnt, neighbors)

    return points_neighbors


def check_neighbors1(type_data, neighbors_idxs, neighbors_dists):

    points_neighbors = OrderedDict()

    radius = {'type 1': R_1, 'type 2A': R_2A, 'type 2X': R_2X,
              'type 1+type 2A': int((R_1+R_2A)/2), 'type 2A+type 2X': int((R_2A+R_2X)/2),
              'type 1+type 2X': int((R_1+R_2X)/2),
              'Null': int((R_1+R_2A+R_2X)/3)}

    # for each point check all their neighbors if they touch the point
    num_points = type_data.shape[0]
    num_neighbors = neighbors_idxs.shape[1]
    for i in range(0, num_points):
        point = type_data[i]
        type = point[4]
        cnt = 0
        neighbors = []
        for j in range(0, num_neighbors):
            # neighbor = type_data[neighbors_idxs[i][j]] - we will need it if will consider to use 'Area' for radius calculating
            distance = neighbors_dists[i][j]
            radiuss = math.sqrt(type_data[neighbors_idxs[i][j]][1]/math.pi) #'area'
            #if 2*radius[type] >= distance:
            if 2 * radiuss * R_koeff_adj >= distance:
                neighbors.append(neighbors_idxs[i][j])
                cnt += 1
            else:
                break
        points_neighbors[i] = (cnt, neighbors)

    return points_neighbors


def calculate_distances(points):
    # distance between two centers
    dist_matrix = dist.cdist(points[..., 2:4], points[..., 2:4])
    return dist_matrix


def pythagorean(x1, x0, y1, y0):
    return (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)


def square_root(number):
    if number < 0:
        print('There is negative argument for sqrt()- ', number)
        number = -number
    return math.sqrt(number)


def dist_btwn_point_line(point, line):

    (line_end1, line_end2) = line
    (x1, y1) = line_end1
    (x2, y2) = line_end2
    (x0, y0) = point

    a2 = pythagorean(x1, x0, y1, y0)
    a = square_root(a2)
    b2 = pythagorean(x2, x0, y2, y0)
    b = square_root(b2)
    l2 = pythagorean(x2, x1, y2, y1)
    l = square_root(l2)

    t1 = (a2 + l2 - b2) / ((2 * a * b) + 0.00001)
    t2 = (b2 + l2 - a2) / ((2 * b * l) + 0.00001)

    if (l2==0) or (t1 < 0) or (t2 < 0):
        r = min(a, b)
    else:
        p = (a + b + l) / 2
        r = (2 / (l + 0.00001)) * square_root(p * (p - l) * (p - a) * (p - b))

    return r


def dist_btwn_two_lines(line1, line2):
    dist = 0

    (line1_end1, line1_end2) = line1
    (line2_end1, line2_end2) = line2

    r0 = dist_btwn_point_line(line2_end1, line1)
    r1 = dist_btwn_point_line(line2_end2, line1)
    r2 = dist_btwn_point_line(line1_end1, line2)
    r3 = dist_btwn_point_line(line1_end2, line2)

    dist = min(r0, r1, r2, r3)

    return dist


def min_dist_btwn_contours(contour1, contour2):

    cntr1 = contour1.reshape(contour1.shape[0], contour1.shape[2])
    cntr2 = contour2.reshape(contour2.shape[0], contour2.shape[2])

    num_points_cntr1 = cntr1.shape[0]
    num_points_cntr2 = cntr2.shape[0]

    dist_array = []
    for i1 in range(0, num_points_cntr1):
        j1 = i1 + 1
        if j1 > num_points_cntr1-1: j1 = 0
        line1 = (cntr1[i1], cntr1[j1])

        #dist_array = []
        for i2 in range(0, num_points_cntr2):
            j2 = i2 + 1
            if j2 > num_points_cntr2 - 1: j2 = 0
            line2 = (cntr2[i2], cntr2[j2])

            dist_array.append(dist_btwn_two_lines(line1, line2))

    min_distance = min(dist_array)
    return min_distance


def min_dist_btwn_contours1(contour1, contour2):
    # min distance between two contours
    min_distance = 0
    cntr1 = contour1.reshape(contour1.shape[0], contour1.shape[2])
    cntr2 = contour2.reshape(contour2.shape[0], contour2.shape[2])
    dist_matrix = dist.cdist(cntr1, cntr2)
    min_distance = dist_matrix.min()

    return min_distance


def find_clusters(type_data, points_neighbors):
    clusters = OrderedDict()

    # sort points by amount of it neighbors
    sorted_neighbors = OrderedDict(sorted(points_neighbors.items(), key=lambda item: item[1][0], reverse=True))

    # start from points with maximum neighbors - recursion???

    # mark the point and it neighbors as a cluster if they reach required minimum number of points
    for i in sorted_neighbors.keys():
        cnt = sorted_neighbors[i][0]
        if cnt >= MIN_NEIGHBORS:

            if type_data[i][6] == 'nan':

                cluster_id_created = False

                cell_set = set(sorted_neighbors[i][1])

                for neighbor in sorted_neighbors[i][1]:

                    if type_data[neighbor][6] == 'nan':

                        neighbor_set = set(sorted_neighbors[neighbor][1])
                        intersection = cell_set.intersection(neighbor_set)

                        if intersection: # there is at least one common neighbor!

                            if not cluster_id_created:
                                global CLUSTER_COUNT
                                CLUSTER_COUNT += 1
                                cluster_id_created = True

                            cluster_idx = CLUSTER_COUNT  # str(uuid.uuid4())[:4]

                            if type_data[i][6] == 'nan':
                                area = type_data[i][1]
                                clusters[cluster_idx] = [1, area, [i]]
                                type_data[i][6] = cluster_idx

                           # if type_data[neighbor][6] == 'nan':
                            type_data[neighbor][6] = cluster_idx
                            clusters[cluster_idx][0] += 1
                            clusters[cluster_idx][1] += type_data[neighbor][1]  # R_1 # should add area!
                            clusters[cluster_idx][2].append(neighbor)

                            mark_cluster(neighbor, cluster_idx, sorted_neighbors, type_data, clusters)

                        else:
                            if type_data[i][6] != 'nan' and sorted_neighbors[neighbor][0] == 1:
                                cluster_idx = type_data[i][6]
                                type_data[neighbor][6] = cluster_idx
                                clusters[cluster_idx][0] += 1
                                clusters[cluster_idx][1] += type_data[neighbor][1]  # R_1 # should add area!
                                clusters[cluster_idx][2].append(neighbor)

    return clusters


def mark_cluster(i, cluster_idx, sorted_neighbors, type_data, clusters):
    (cnt, neighbors) = sorted_neighbors[i]

    cell_set = set(sorted_neighbors[i][1])

    for neighbor_idx in neighbors:
         if type_data[neighbor_idx][6] == 'nan':

            # if this neighbor has no other neighbors but i-th
            if sorted_neighbors[neighbor_idx][0] == 1:
                type_data[neighbor_idx][6] = cluster_idx

                clusters[cluster_idx][0] += 1
                clusters[cluster_idx][1] += type_data[neighbor_idx][1]  # R_1 # should add area!
                clusters[cluster_idx][2].append(neighbor_idx)

            else: # check the list of its neighbors if there are any intersections with i's neighbors

                neighbor_set = set(sorted_neighbors[neighbor_idx][1])
                intersection = cell_set.intersection(neighbor_set)

                if intersection:  # there is at least one common neighbor!
                    type_data[neighbor_idx][6] = cluster_idx

                    clusters[cluster_idx][0] += 1
                    clusters[cluster_idx][1] += type_data[neighbor_idx][1]  # R_1 # should add area!
                    clusters[cluster_idx][2].append(neighbor_idx)

                    # going deeper
                    mark_cluster(neighbor_idx, cluster_idx, sorted_neighbors, type_data, clusters)


def find_clusters2(type_data, points_neighbors):
    clusters = OrderedDict()

    # sort points by amount of it neighbors
    sorted_neighbors = OrderedDict(sorted(points_neighbors.items(), key=lambda item: item[1][0], reverse=True))

    # start from points with maximum neighbors - recursion???

    # mark the point and it neighbors as a cluster if they reach required minimum number of points
    for i in sorted_neighbors.keys():
        cnt = sorted_neighbors[i][0]
        if cnt >= MIN_NEIGHBORS:

            if type_data[i][6] == 'nan':

                # cluster_id_created = False # modification 03-30-2020:

                # modification 03-30-2020:
                # if cell has more than minimum required neighbors, they are considered as a cluster all together,
                # even if they do not connect to each other
                global CLUSTER_COUNT
                CLUSTER_COUNT += 1

                cluster_idx = CLUSTER_COUNT  # str(uuid.uuid4())[:4]

                area = type_data[i][1]
                clusters[cluster_idx] = [1, area, [i]]
                type_data[i][6] = cluster_idx

                cell_set = set(sorted_neighbors[i][1])

                for neighbor in sorted_neighbors[i][1]:

                    if type_data[neighbor][6] == 'nan':

                        # cluster_idx = type_data[i][6]
                        type_data[neighbor][6] = cluster_idx
                        clusters[cluster_idx][0] += 1
                        clusters[cluster_idx][1] += type_data[neighbor][1]  # R_1 # should add area!
                        clusters[cluster_idx][2].append(neighbor)

                        neighbor_set = set(sorted_neighbors[neighbor][1])
                        intersection = cell_set.intersection(neighbor_set)

                        if intersection: # there is at least one common neighbor!

                            # modification 03-30-2020:
                            # if not cluster_id_created:
                            #     global CLUSTER_COUNT
                            #    CLUSTER_COUNT += 1
                            #    cluster_id_created = True

                            # cluster_idx = CLUSTER_COUNT  # str(uuid.uuid4())[:4]

                            # modification 03-30-2020:
                            # if type_data[i][6] == 'nan':
                            #    area = type_data[i][1]
                            #    clusters[cluster_idx] = [1, area, [i]]
                            #    type_data[i][6] = cluster_idx

                           # if type_data[neighbor][6] == 'nan':
                            # modification 03-30-2020:
                            #type_data[neighbor][6] = cluster_idx
                            #clusters[cluster_idx][0] += 1
                            #clusters[cluster_idx][1] += type_data[neighbor][1]  # R_1 # should add area!
                            #clusters[cluster_idx][2].append(neighbor)

                            mark_cluster2(neighbor, cluster_idx, sorted_neighbors, type_data, clusters)

                        # modification 03-30-2020:
                        # else:
                        #    if type_data[i][6] != 'nan' and sorted_neighbors[neighbor][0] == 1:
                        #        cluster_idx = type_data[i][6]
                        #        type_data[neighbor][6] = cluster_idx
                        #        clusters[cluster_idx][0] += 1
                        #        clusters[cluster_idx][1] += type_data[neighbor][1]  # R_1 # should add area!
                        #        clusters[cluster_idx][2].append(neighbor)

        cluster_idx = type_data[i][6]
        if cluster_idx in clusters.keys():
            if clusters[cluster_idx][0] <= 1:
                del clusters[cluster_idx]

    return clusters


def mark_cluster2(i, cluster_idx, sorted_neighbors, type_data, clusters):
    (cnt, neighbors) = sorted_neighbors[i]

    cell_set = set(sorted_neighbors[i][1])

    for neighbor_idx in neighbors:
         if type_data[neighbor_idx][6] == 'nan':

            # if this neighbor has no other neighbors but i-th
            if sorted_neighbors[neighbor_idx][0] == 1:
                type_data[neighbor_idx][6] = cluster_idx

                clusters[cluster_idx][0] += 1
                clusters[cluster_idx][1] += type_data[neighbor_idx][1]  # R_1 # should add area!
                clusters[cluster_idx][2].append(neighbor_idx)

            else: # check the list of its neighbors if there are any intersections with i's neighbors

                neighbor_set = set(sorted_neighbors[neighbor_idx][1])
                intersection = cell_set.intersection(neighbor_set)

                if intersection:  # there is at least one common neighbor!
                    type_data[neighbor_idx][6] = cluster_idx

                    clusters[cluster_idx][0] += 1
                    clusters[cluster_idx][1] += type_data[neighbor_idx][1]  # R_1 # should add area!
                    clusters[cluster_idx][2].append(neighbor_idx)

                    # going deeper
                    mark_cluster2(neighbor_idx, cluster_idx, sorted_neighbors, type_data, clusters)


def clustering(groupedListData, contours):
    max_number_of_checking_neighbors = MAX_NEIGHBORS + 1  # add 1 because one of them is always the point itself

    # groups = grouped_data_by_type(data_frame)
    # type1, type2A, type2X = grouped_data_by_type(data_frame)
    clusters_grouped = []

    for (type_group, type) in groupedListData:

        if len(type_group) == 0:
            continue

        dist_matrix = calculate_distances(type_group)
        rows_sort = dist_matrix.argsort(axis=1)[..., 1:max_number_of_checking_neighbors]
        dist_sorted = np.sort(dist_matrix, axis=1)[..., 1:max_number_of_checking_neighbors]

        points_neighbors = check_neighbors(type_group, rows_sort, dist_sorted, contours)

        # adding a field for cluster information
        type_group = np.insert(type_group, 6, 'nan', axis=1)
        clusters = find_clusters(type_group, points_neighbors)

        clusters_grouped.append((clusters, type, type_group))

    write_clusters_to_csv(clusters_grouped)

    return clusters_grouped


def write_clusters_to_csv(clusters_grouped):
    with open(clusters_filename, mode='w') as clusters_file:
        clusters_writer = csv.writer(clusters_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for (clusters, type, data) in clusters_grouped:

            clusters_writer.writerow(['Type', 'Total'])
            clusters_writer.writerow([type, len(clusters)])

            clusters_writer.writerow(['ClusterID', 'Type', 'Size', 'Area', 'Members'])

            for key in clusters:
                (cnt, area, members) = clusters[key]
                members_ids = []
                for idx in members:
                    members_ids.append(data[idx][0])

                clusters_writer.writerow([key, type, cnt, area, members_ids])


if __name__ == '__main__':
    fig, ax = plt.subplots()

    contours = open_contours(contours_file)
    # plot_contours(contours) # for testing input data
    data = open_data(file_name)
    plot_pandasData(data, ax)
    data_and_contours = connect_points_with_contours(data, contours)

    groupedListData = grouped_data_by_type(data_and_contours)
    plot_listData(groupedListData, contours, ax)

    clusters = clustering(groupedListData, contours)
    plot_clusters(clusters, ax)

    plt.legend()
    plt.title(cell_name + all)
    plt.show()
