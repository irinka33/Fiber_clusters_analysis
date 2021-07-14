# Fall 2020: new type annotation:
# type 1 = 1
# type 2A = 2
# type 2X = 3
# type 1 + type 2A = 12
# type 1 + type 2X = 13
# type 2A + type 2X = 23
# unknown ('Null') = 0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import cv2
import csv
import math
import xml.etree.ElementTree as ET
from datetime import date
import os

# INPUT
# select the mode: MIX_ON = True for mix double typed fibrils with two corresponding main types
#                  MIX_ON = False  for analysing double typed fibrils as independent typed
MIX_ON = True
USE_UNKNOWN = False
frame_w = 2.78
frame_h = 3.77
# (12.37, 16.39)
fig_size = (20.26, 12.41) # (15.7 + frame_w, 9.55 + frame_h) #(20, 25)  # размер картинки в дюймах

cell_name = 'H_SS_02_p1'  # - 1.25 and min dist = 20

# 'H_SS_01_p1'  # - 1.25 and min dist = 20!
# 'H_SS_02_p1'  # - 1.25 and min dist = 20
# 'H_SS_02_p2'  # - 1.25 and min dist = 20
# 'H_SS_04_p2'  # - 1.25 and min dist = 20
# 'H_SS_04_p3'  # - 1.25 and min dist = 25
# 'H_SS_06_p1'  # - 1.25 and min dist = 30
# 'H_SS_06_p2'  # - 1.25 and min dist = 30
# 'H_SS_06_p3'  # - 1.25 and min dist = 40
# 'H_SS_09_p2B'  # - 0.3126 and min dist = 20
# 'H_SS_10_p3'  # - 1.25 and min dist = 50
# 'H_SS_11_p1'  # - 1.25 and min dist = 20
# 'H_SS_11_p2'  # - 1.25 and min dist = 20
# 'H_SS_14_p3'  # - 1.25 and min dist = 20
# 'H_SS_17_p2B'  # - 0.3126 and min dist = 25???
# 'H_SS_18_p2A'  # - 0.3126 - переделать под картинку размера (25,20) - долго - and min dist = 20
# 'H_SS_18_p3A'  # - 1.25 and min dist = 20!
# 'H_SS_19_p1'  # - 1.25 and min dist = 20
# 'H_SS_21_p2B'  # - 0.3126 and min dist = 20!
# 'H_SS_21_p2A'  # - 0.3126 and min dist = 20! (делала 10 - хуже)
# 'H_SS_21_p1A'  # - 0.3126 and min dist = 20!
# 'H_SS_30_p3B'  # - 1.25 and min dist = 20
# 'H_SS_30_p1'  # - 1.25 and min dist = 20
# 'H_SS_27_p3'  # - 1.25 and min dist = 20
# 'H_SS_27_p1'  # - 1.25 and min dist = 20
# 'H_SS_26_p1A'  # - 0.3126 and min dist = 20
# 'H_SS_24_p1'  # -1 and min dist = 20!
# 'H_SS_23_p4B'  # - 0.3126 and min dist = 25
# 'H_SS_23_p4A'  # - 0.3126 - сделал много вариантов от 10 до 40 - дальше уже нельзя - выбрать!
# 'H_SS_31_p3'  # - 1.25 and min dist = 20
# 'H_SS_31_p2'  # - 1.25 and min dist = 20
# 'H_SS_31_p1'  # - 1.25 and min dist = 20 and fig_size = (6.27, 11.78) #(9.59, 12.62333)
# 'H_SS_22_p3'  # - 1 and min dist = 20
# 'H_SS_13_p2'  # - 1.0 and min dist = 22
# 'H_SS_08_p3'  # -0.3126 - сделала еще вариант с меньшим расстоянием=10! - лучше!
# 'H_SS_08_p2'  # - 1.25 and min dist = 24.2!
# 'H_SS_03_p3'  # - 1.25 + warnings and min dist = 20
# 'H_SS_07_p1'  # - 1.25 and min dist = 20!
# 'H_SS_20_p3'  # - 1.25 and min dist = 20


scale =  1.25 # 0.3126 # 1.25 - берем из файлв cz
camera_dist = 12.5  # - берем из файла cz - что делать, если нету???

WARNINGS = []

R_1 = 39.5
R_2A = 38
R_2X = 35
MAX_NEIGHBORS = 10
MIN_POINTS_PER_CLUSTER = 5
MIN_NEIGHBORS = 4
MIN_DISTANCE_BETWEEN_CONTOURS = 20 #30 #25 # 2*(camera_dist/scale) # 12.5 um in pixels - берем из файла cz

CLUSTER_COUNT = 0

R_koeff_adj = 1  # adjusting coefficient for radius -
# should be calculated as weighted combination of minEnclosedCircle's radius and radius from 'area'
# Input csv file format:
#  'ElementID', 'Area', 'CenterX', 'CenterY', 'type',
# Fields leaving/adding for processing:
#  'ElementID', 'Area', 'CenterX', 'CenterY', 'type', 'contour_id', 'cluster_id'
#       0          1         2         3         4          5           6

dir_name = 'data\\raw_data\\' + cell_name
file_name = dir_name + '\\input' + cell_name + '.csv'
# image_name = 'data\input\imageSS01.j pg'
contours_file = dir_name + '\\contours' + cell_name + '.cz'

# OUTPUT
if MIX_ON:
    fl_all = '_all_'
else:
    fl_all = '_'

# clusters_dir = 'data\\output\\csv\\' + str(date.today()) + cell_name + '\\'
clusters_dir = 'data\\output\\' + str(date.today()) + '\\' + cell_name + '\\'
if not os.path.exists(clusters_dir):
    os.makedirs(clusters_dir)

clusters_filename = clusters_dir + 'clusters' + cell_name + fl_all + str(date.today()) + '_' + \
                    str(MIN_DISTANCE_BETWEEN_CONTOURS) + '.csv'
points_filename = clusters_dir + 'points_' + cell_name + fl_all + str(date.today()) + '_' + \
                  str(MIN_DISTANCE_BETWEEN_CONTOURS) + '.csv'
fig_name = clusters_dir + 'clusters' + cell_name + fl_all + str(date.today()) + '_' + \
           str(MIN_DISTANCE_BETWEEN_CONTOURS) + '.pdf'


def open_contours(contours_file):
    contours = []

    if os.path.exists(contours_file):

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
    else:
        print('Input contours file does not exist')

    return contours


def plot_contours(contours):
    # for testing propose
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

    if os.path.exists(file_name):
        pandasData = pd.read_csv(file_name,
                           names=['ElementID', 'Area',
                                  'CenterX', 'CenterY', 'type'], skiprows=1)

        # TODO - решить, где делать массшатбирование - здесь или в контурах, сейчас делается в контурах
        # pandasData['CenterX'] /= scale #0.31 #1.25 #pandasData['CenterX'].real
        # pandasData['CenterY'] /= scale # 0.31 #1.25
        # pandasData['type'] = pandasData['type'].str.strip()
        pandasData['type'].fillna(0, inplace=True)
        pandasData['type'] = pandasData['type'].astype(int)

    else:
        print('Input data file does not exist')
        pandasData = []

    return pandasData


def plot_pandasData(pandasData, ax):
    D1 = R_1 # * 2
    D2 = R_2A # * 2
    D3 = R_2X # * 2
    colors = {1: 'y', 2: 'r', 3: 'fuchsia',
              12: 'b', 23: 'purple', 13: 'orange',
              0: 'black'}

    size = {1: D1, 2: D2, 3: D3,
            12: int((D1 + D2) / 2), 23: int((D2 + D3) / 2),
            13: int((D1 + D3) / 2),
            0: int((D1 + D2 + D3) / 3)}

    pandasData_grouped = pandasData.groupby('type')

    for key, group in pandasData_grouped:
        group.plot(x='CenterX', y='CenterY', ax=ax, kind='scatter', subplots=True,
                   # title='input data', label=key, c=colors[key], s=size[key] * R_koeff_adj, alpha=0.65, legend=True)
                    title = None, label = key, c = colors[key], s = size[key] * R_koeff_adj, alpha = 0.65, legend = True)

def connect_points_with_contours(pandasData, contours):

    cx = np.array([pandasData['CenterX'].to_numpy()])
    cy = np.array([pandasData['CenterY'].to_numpy()])
    points_centers = np.append(cx, cy, axis = 0).transpose()

    contours_centers = []
    for i, contour in enumerate(contours):
        M = cv2.moments(contour)

        if M['m00'] == 0:
            cx = contour[0][0][0]
            cy = contour[0][0][1]
            warning = 'There is a contour with less than 4 points, ' + str(i) + str(contour.tolist())
            print(warning)
                #'There is a contour with less than 4 points, ', i, contour.tolist())
            global WARNINGS
            WARNINGS.append(warning)
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
        warning = 'Data and contours do not match:' + '\n' + \
                  '     for points ' + str(unusedRows) + '\n' +\
                  '     for contours ' + str(unusedCols)
        #print('Data and contours do not match:')
        #print('     for points ', unusedRows)
        #print('     for contours ', unusedCols)
        print(warning)
        #global WARNINGS
        WARNINGS.append(warning)

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

    separate_semi_types = True

    for type, group in rowData_grouped:
        # print(group.head())
        type_gr = group.to_numpy()
        if type == 1:
            if len(type1_points) == 0: type1_points = type_gr
            else: type1_points = np.append(type1_points, type_gr, axis=0)
        elif type == 2:
            if len(type2A_points) == 0: type2A_points = type_gr
            else: type2A_points = np.append(type2A_points, type_gr, axis=0)
        elif type == 3:
            if len(type2X_points) == 0: type2X_points = type_gr
            else: type2X_points = np.append(type2X_points, type_gr, axis=0)

        #---
        elif type == 12:
            if not separate_semi_types:
                if len(type1_points) == 0: type1_points = type_gr
                else: type1_points = np.append(type1_points, type_gr, axis=0)
                if len(type2A_points) == 0: type2A_points = type_gr
                else: type2A_points = np.append(type2A_points, type_gr, axis=0)
            else:
                if len(type1type2A_points) == 0: type1type2A_points = type_gr
                else: type1type2A_points = np.append(type1type2A_points, type_gr, axis=0)
        elif type == 23:
            if not separate_semi_types:
                if len(type2A_points) == 0: type2A_points = type_gr
                else: type2A_points = np.append(type2A_points, type_gr, axis=0)
                if len(type2X_points) == 0: type2X_points = type_gr
                else: type2X_points = np.append(type2X_points, type_gr, axis=0)
            else:
                if len(type2Atype2X_points) == 0: type2Atype2X_points = type_gr
                else: type2Atype2X_points = np.append(type2Atype2X_points, type_gr, axis=0)
        elif type == 13:
            if not separate_semi_types:
                if len(type1_points) == 0: type1_points = type_gr
                else: type1_points = np.append(type1_points, type_gr, axis=0)
                if len(type2X_points) == 0: type2X_points = type_gr
                else: type2X_points = np.append(type2X_points, type_gr, axis=0)
            else:
                if len(type1type2X_points) == 0: type1type2X_points = type_gr
                else: type1type2X_points = np.append(type1type2X_points, type_gr, axis=0)
        else: # no type (type == Null)
            if not separate_semi_types:
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
    if not separate_semi_types:
        return (type1_points, 1), (type2A_points, 2), (type2X_points, 3), \
               (unknown_type_points, 0)
    else:
        return (type1_points, 1), (type2A_points, 2), (type2X_points, 3), \
               (type1type2A_points, 12), (type1type2X_points, 13), \
               (type2Atype2X_points, 23),\
               (unknown_type_points, 0)


def plot_listData(data_grouped, contours, ax):
    colors = {1: 'y', 2: 'r', 3: 'fuchsia',
              12: 'b', 23: 'purple', 13: 'orange',
              0: 'black'}

    # fig, ax = plt.subplots()
    for (group, type) in data_grouped:
        for point in group:
            x = point[2]
            y = point[3]
            ax.plot(x, y, color=colors[type], alpha=0.45, marker='s', markersize=5)

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


def plot_clusters(data, clusters, ax):
    D1 = R_1*2#20
    D2 = R_2A*2#25
    D3 = R_2X*2#30

    # fig, ax = plt.subplots()
    colors = {1: 'y', 2: 'r', 3: 'fuchsia',
              12: 'b', 23: 'purple', 13: 'orange',
              0: 'black'}

    for (cluster_group, cluster_type) in clusters:
        for (key, cnt, area, members) in cluster_group:
            x =[]
            y =[]
            #(cnt, area, members) = cluster_group[key]
            for idx in members:
                cx = data['CenterX'][idx]
                cy = data['CenterY'][idx]
                x.append(cx)
                y.append(cy)
                ax.text(cx-3, cy-3, key)
            ax.plot(x, y, color=colors[cluster_type], alpha=0.45, linestyle='-', marker='X', markersize=8)
            # ax.plot(x, y, color=colors[cluster_type], alpha=0.45, linestyle='', marker='X', markersize=8)


def same_type(point_type,neighbor_type):
    same = False
    type = 0

    if point_type == neighbor_type:
        same = True
        type = point_type

    else:
        if  MIX_ON:
            str_point_type = str(int(point_type))
            str_neighbor_type = str(int(neighbor_type))

            if (str_point_type in str_neighbor_type) or (str_neighbor_type in str_point_type):
                same = True
                type = point_type if len(str_point_type)==1 else neighbor_type

    return (same, type)


def check_neighbors_for_contacts(points_with_neighbors, contours, data):

    points_contact_neighbors = OrderedDict()

    # for each point check all their neighbors if they touch the point
    dist_matrix = np.ones((data.shape[0], data.shape[0]))*(-1)

    for (point_idx, neighbors_idxs) in points_with_neighbors:
        point = data[point_idx]
        point_type = point[4]
        contour_id = int(point[5])

        # TODO: if there is no contours for current point - we can use distance between centers???
        # it skips this point for now
        if contour_id == -1:
            continue
        point_contour = contours[contour_id]

        cnt = 0
        neighbors = []
        for neighbor in neighbors_idxs:

            contour_id = int(data[neighbor][5])
            # TODO: if there is no contours for neighbor point - we can use distance between centers???
            # it skips this neighbor for now
            if contour_id == -1:
                continue
            neighbor_contour = contours[contour_id]

            if dist_matrix[point_idx][neighbor] == -1:
                min_dist = min_dist_btwn_contours(point_contour, neighbor_contour)
                dist_matrix[point_idx][neighbor] = min_dist
                dist_matrix[neighbor][point_idx] = min_dist
            else:
                min_dist = dist_matrix[point_idx][neighbor]

            if min_dist <= MIN_DISTANCE_BETWEEN_CONTOURS:
                # checking cell's types
                neighbor_type = data[neighbor][4]
                if same_type(point_type,neighbor_type)[0]:
                    neighbors.append(neighbor)
                    cnt += 1
                else:
                    cnt = 0
                    neighbors = []
                    # stop checking neighbors for this point - it is not in our interest anymore
                    break
        if cnt != 0:
            points_contact_neighbors[point_idx] = (cnt, neighbors)

    return points_contact_neighbors


def check_all_neighbors_for_contacts(points_with_neighbors, contours, data):

    points_contact_neighbors = OrderedDict()

    # for each point check all their neighbors if they touch the point
    dist_matrix = np.ones((data.shape[0], data.shape[0]))*(-1)

    for (point_idx, neighbors_idxs) in points_with_neighbors:
        point = data[point_idx]
        contour_id = int(point[5])

        # TODO: if there is no contours for current point - we can use distance between centers???
        # it skips this point for now
        if contour_id == -1:
            continue
        point_contour = contours[contour_id]

        cnt = 0
        neighbors = []
        for neighbor in neighbors_idxs:

            contour_id = int(data[neighbor][5])
            # TODO: if there is no contours for neighbor point - we can use distance between centers???
            # it skips this neighbor for now
            if contour_id == -1:
                continue
            neighbor_contour = contours[contour_id]

            if dist_matrix[point_idx][neighbor] == -1:
                min_dist = min_dist_btwn_contours(point_contour, neighbor_contour)
                dist_matrix[point_idx][neighbor] = min_dist
                dist_matrix[neighbor][point_idx] = min_dist
            else:
                min_dist = dist_matrix[point_idx][neighbor]

            if min_dist <= MIN_DISTANCE_BETWEEN_CONTOURS:
                neighbors.append(neighbor)
                cnt += 1

        if cnt >= 5:
            points_contact_neighbors[point_idx] = (cnt, neighbors)

    return points_contact_neighbors


def nearest_neighbors_by_distances(points, max_number_of_neighbors):
    # distance between two centers

    dist_matrix = dist.cdist(points[..., 2:4], points[..., 2:4])
    rows_sort = dist_matrix.argsort(axis=1)[..., 1:max_number_of_neighbors]
    dist_sorted = np.sort(dist_matrix, axis=1)[..., 1:max_number_of_neighbors]

    return rows_sort, dist_sorted


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


def primary_type(type1, type2):
    if type1 == type2:
        return type1

    else:
        if (str(int(type1)) in str(int(type2))) or (str(int(type2)) in str(int(type1))):
            type = type1 if len(str(int(type1))) == 1 else type2
            return type
        else:
            return 0 # ERROR!


def adsorb_cluster_with_primary_types(new_cluster_id, old_cluster_id, clusters, type_data):
    if new_cluster_id == old_cluster_id:
        return

    # type = same_type(clusters[new_cluster_id][3], clusters[old_cluster_id][3])[1]

    # clusters[new_cluster_id][0] += clusters[old_cluster_id][0]
    # clusters[new_cluster_id][1] += clusters[old_cluster_id][1]

    new_list = list(set(clusters[new_cluster_id][2] + clusters[old_cluster_id][2]))
    clusters[new_cluster_id][2] = new_list
    clusters[new_cluster_id][0] = len(new_list)
    # clusters[new_cluster_id][3] = type

    for member in clusters[old_cluster_id][2]:
        type_data[member][6] = new_cluster_id

    del clusters[old_cluster_id]


def adsorb_cluster_with_double_semi_types(new_cluster_id, old_cluster_id, clusters, type_data):
    if new_cluster_id == old_cluster_id:
        return

    # type = same_type(clusters[new_cluster_id][3], clusters[old_cluster_id][3])[1]

    # clusters[new_cluster_id][0] += clusters[old_cluster_id][0]
    # clusters[new_cluster_id][1] += clusters[old_cluster_id][1]

    new_list = list(set(clusters[new_cluster_id][2] + clusters[old_cluster_id][2]))
    clusters[new_cluster_id][2] = new_list
    clusters[new_cluster_id][0] = len(new_list)
    # clusters[new_cluster_id][3] = type

    for member in clusters[old_cluster_id][2]:
        type_data[member][6] = new_cluster_id

    del clusters[old_cluster_id]

    '''
    type = str(clusters[old_cluster_id][3])
    # do not delete semi-clusters
    if len(type) == 1:
        del clusters[old_cluster_id]
    else:
        type = str(clusters[old_cluster_id][3])
    '''


def determine_cluster_type(sorted_neighbors, data):
    neighbors_type = {0: 0, 1: 0, 2: 0, 3: 0 , 12: 0, 13: 0, 23: 0}
    confirmed_neighbors = []

    for neighbor in sorted_neighbors:
        neighbors_type[data[neighbor][4]] += 1

    cluster_type = max(neighbors_type, key=neighbors_type.get)
    neighbors_type = OrderedDict(sorted(neighbors_type.items(), key=lambda item: item[1], reverse=True))

    for key, value in neighbors_type.items():
        if len(str(int(key))) == 1:
            cluster_type = key
            break

    # checking all neighbors for types conflict
    for neighbor in sorted_neighbors:
        if same_type(cluster_type, data[neighbor][4])[0]:
            confirmed_neighbors.append(neighbor)

    return (list(confirmed_neighbors), cluster_type)


def find_clusters_with_primary_types_privilege(points_neighbors, contours, data):
    clusters = OrderedDict()
    clusters_grouped = OrderedDict()

    # sort points by amount of it neighbors
    sorted_neighbors = OrderedDict(sorted(points_neighbors.items(), key=lambda item: item[1][0], reverse=True))

    # start from points with maximum neighbors - recursion???

    # mark the point and it neighbors as a cluster if they reach required minimum number of points
    for i in sorted_neighbors.keys():
        cnt = sorted_neighbors[i][0]
        if cnt >= MIN_NEIGHBORS:

            # create a new cluster
            global CLUSTER_COUNT
            CLUSTER_COUNT += 1
            new_cluster_id = CLUSTER_COUNT

            area = 0 #type_data[i][1]
            type = data[i][4]
            clusters[new_cluster_id] = [1, area, [i], type]

            # if there is any connected cells here that is already a member of an existing cluster,
            # new cluster adsorbs this existing cluster
            if not np.isnan(data[i][6]):
            # if data[i][6] != 'nan':
                cluster_id = data[i][6]
                old_cluster_type = clusters[cluster_id][3]
                clusters[new_cluster_id][3] = primary_type(type, old_cluster_type)
                adsorb_cluster_with_primary_types(new_cluster_id, cluster_id, clusters, data)

            data[i][6] = new_cluster_id

            for neighbor in sorted_neighbors[i][1]:
                cluster_id = data[neighbor][6]

                # we need type of neighbor to change the cluster type if this new type is one of primary
                neighbor_type = data[neighbor][4]

                if not np.isnan(cluster_id):
                # if cluster_id != 'nan':
                    # we need type of old cluster to change the cluster type if this new type is one of primary
                    if cluster_id != new_cluster_id:
                        old_cluster_type = clusters[cluster_id][3]
                        clusters[new_cluster_id][3] = primary_type(type, old_cluster_type)
                        adsorb_cluster_with_primary_types(new_cluster_id, cluster_id, clusters, data)
                else:
                    clusters[new_cluster_id][3] = primary_type(type, neighbor_type)
                    clusters[new_cluster_id][0] += 1
                    #clusters[new_cluster_id][1] += type_data[neighbor][1]
                    clusters[new_cluster_id][2].append(neighbor)

                data[neighbor][6] = new_cluster_id
        else:
            return clusters

    return clusters


def find_clusters_with_double_enter_semi_types(points_neighbors, contours, data):
    clusters = OrderedDict()

    # sort points by amount of it neighbors
    sorted_neighbors = OrderedDict(sorted(points_neighbors.items(), key=lambda item: item[1][0], reverse=True))

    # start from points with maximum neighbors - recursion???

    # mark the point and it neighbors as a cluster if they reach required minimum number of points
    for i in sorted_neighbors.keys():
        cnt = sorted_neighbors[i][0]
        if cnt >= MIN_NEIGHBORS:

            # determine cluster type:
            # for primary types it is the same type,
            # for semi-types - check the type of most fibrils
            cluster_type = data[i][4]

            if len(str(int(cluster_type))) != 1:
                confirmed_neighbors, cluster_type = determine_cluster_type(sorted_neighbors[i][1], data)
                # sorted_neighbors[i][1] = confirmed_neighbors
                conf_cnt = len(confirmed_neighbors)
            else:
                confirmed_neighbors = sorted_neighbors[i][1]
                conf_cnt = cnt

            # if cnt < MIN_NEIGHBORS:
            if conf_cnt != cnt:
                continue

            # create a new cluster
            global CLUSTER_COUNT
            CLUSTER_COUNT += 1
            new_cluster_id = CLUSTER_COUNT

            area = 0  # type_data[i][1]
            clusters[new_cluster_id] = [1, area, [i], cluster_type]

            # if there is any connected cells here that is already a member of an existing cluster,
            # new cluster adsorbs this existing cluster
            if not np.isnan(data[i][6]):
                # if data[i][6] != 'nan':
                cluster_id = data[i][6]
                old_cluster = same_type(cluster_type, clusters[cluster_id][3])
                if old_cluster[0]:
                    cluster_type = old_cluster[1]
                    clusters[new_cluster_id][3] = cluster_type
                    adsorb_cluster_with_double_semi_types(new_cluster_id, cluster_id, clusters, data)
                else:
                    continue # next cluster base

            data[i][6] = new_cluster_id

            for neighbor in confirmed_neighbors:
            # for neighbor in sorted_neighbors[i][1]:
                cluster_id = data[neighbor][6]

                if not np.isnan(cluster_id):
                    # if cluster_id != 'nan':
                    # we need type of old cluster to change the cluster type if this new type is one of primary
                    old_cluster = same_type(cluster_type, clusters[cluster_id][3])
                    if old_cluster[0]:
                        cluster_type = old_cluster[1]
                        clusters[new_cluster_id][3] = cluster_type
                        adsorb_cluster_with_double_semi_types(new_cluster_id, cluster_id, clusters, data)
                    else:
                        continue # next neighbor

                else:
                    clusters[new_cluster_id][0] += 1
                    # clusters[new_cluster_id][1] += type_data[neighbor][1]
                    clusters[new_cluster_id][2].append(neighbor)

                data[neighbor][6] = new_cluster_id

            if clusters[new_cluster_id][0] < MIN_NEIGHBORS:
                for member in clusters[new_cluster_id][2]:
                    data[member][6] = float('nan')

                del clusters[new_cluster_id]

        else:
            return clusters

    return clusters


def mark_cluster(i, cluster_idx, sorted_neighbors, type_data, clusters):
    (cnt, neighbors) = sorted_neighbors[i]

    cell_set = set(sorted_neighbors[i][1])

    for neighbor_idx in neighbors:
         if np.isnan(type_data[neighbor_idx][6]):
         # if type_data[neighbor_idx][6] == 'nan':

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


def mark_cluster2(i, cluster_idx, sorted_neighbors, type_data, clusters):
    (cnt, neighbors) = sorted_neighbors[i]

    cell_set = set(sorted_neighbors[i][1])

    for neighbor_idx in neighbors:
         if np.isnan(type_data[neighbor_idx][6]):
         # if type_data[neighbor_idx][6] == 'nan':

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


def group_clusters_by_type(clusters):

    type1 = []
    type2A = []
    type2X = []

    type1type2A = []
    type2Atype2X = []
    type1type2X = []
    unknown_type = []

    for id in clusters.keys():
        type = clusters[id][3]
        member = (id, clusters[id][0], clusters[id][1], clusters[id][2])

        if type == 1:
            type1.append(member)
        elif type == 2:
            type2A.append(member)
        elif type == 3:
            type2X.append(member)

        # ---
        elif type == 12:
            type1type2A.append(member)
        elif type == 23:
            type2Atype2X.append(member)
        elif type == 12:
            type1type2X.append(member)
        else:  # no type (type == Null)
            unknown_type.append(member)

    return ((type1, 1), (type2A, 2), (type2X, 3),
            (type1type2A, 12), (type1type2X, 13),
            (type2Atype2X, 23), (unknown_type, 0))


def clustering(pandas_data, contours):
    max_number_of_checking_neighbors = MAX_NEIGHBORS + 1  # add 1 because one of them is always the point itself

    data = pandas_data.to_numpy()
    neighbors_idx, dist_sorted = nearest_neighbors_by_distances(data, max_number_of_checking_neighbors)

    num_points = data.shape[0]
    # num_neighbors = neighbors_idx.shape[1]
    points_neighbors = []
    for i in range(0, num_points):
        point_cnt = 0
        for neighbor in neighbors_idx[i]:
            # if data[i][4] == data[neighbor][4]:
            if same_type(pandas_data['type'][i], pandas_data['type'][neighbor])[0]:
                point_cnt += 1
        if point_cnt >= MIN_NEIGHBORS:
            points_neighbors.append((i, neighbors_idx[i]))

    points_contact_neighbors = check_neighbors_for_contacts(points_neighbors, contours, data)

    # adding a field for cluster information
    data = np.insert(data, 6, float('nan'), axis=1)
    # clusters = find_clusters_with_primary_types_privilege(points_contact_neighbors, contours, data)
    clusters = find_clusters_with_double_enter_semi_types(points_contact_neighbors, contours, data)

    clusters_grouped = group_clusters_by_type(clusters)
    write_clusters_to_csv(clusters_grouped, data)

    return clusters_grouped


def write_clusters_to_csv(clusters_grouped, data):
    with open(clusters_filename, mode='w') as clusters_file:
        clusters_writer = csv.writer(clusters_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for (clusters, cluster_type) in clusters_grouped:

            clusters_writer.writerow(['Type', 'Total'])
            clusters_writer.writerow([int(cluster_type), len(clusters)])

            if len(clusters) == 0:
                continue

            clusters_writer.writerow(['ClusterID', 'Type', 'Size', 'Area', 'Members'])

            for (key, cnt, area, members) in clusters:
                # (cnt, area, members) = clusters[key]
                area = 0
                members_ids = []
                for idx in members:
                    members_ids.append(int(data[idx][0]))
                    area += data[idx][1]

                clusters_writer.writerow([int(key), int(cluster_type), int(cnt), area, members_ids])

        # for warning in WARNINGS:
        clusters_writer.writerow(WARNINGS)


def clustering_for_info(pandas_data, contours):
    max_number_of_checking_neighbors = MAX_NEIGHBORS + 1  # add 1 because one of them is always the point itself

    data = pandas_data.to_numpy()
    neighbors_idx, dist_sorted = nearest_neighbors_by_distances(data, max_number_of_checking_neighbors)

    num_points = data.shape[0]
    # num_neighbors = neighbors_idx.shape[1]
    points_neighbors = []
    for i in range(0, num_points):
        points_neighbors.append((i, neighbors_idx[i]))

    points_contact_neighbors = check_all_neighbors_for_contacts(points_neighbors, contours, data)

    write_neighbors_info_to_csv(points_contact_neighbors, data)


def write_neighbors_info_to_csv(points_contact_neighbors, data):
    with open(points_filename, mode='w') as points_file:
        points_writer = csv.writer(points_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        points_writer.writerow(['ID', 'Type', 'Number of neighbors', 'Neighbor ID', 'Neighbor Type'])

        if len(points_contact_neighbors) == 0:
            return

        for point_idx in points_contact_neighbors.keys():
            (cnt, neighbors_ids) = points_contact_neighbors[point_idx]

            # points_writer.writerow([int(point_idx), int(data[point_idx][4]), int(cnt), 'ID', 'Type'])
            points_writer.writerow([int(data[point_idx][0]), int(data[point_idx][4]), int(cnt), 'ID', 'Type'])

            for neighbor in neighbors_ids:
                # points_writer.writerow(['', '', '', neighbor, data[neighbor][4]])
                points_writer.writerow(['', '', '', int(data[neighbor][0]), int(data[neighbor][4])])


def info_for_all_neighbors():

    contours = open_contours(contours_file)
    if len(contours) == 0:
        return

    data = open_data(file_name)
    if len(data) == 0:
        return

    data_and_contours = connect_points_with_contours(data, contours)
    if len(data_and_contours) == 0:
        return

    grouped_list_data = grouped_data_by_type(data_and_contours)
    clustering_for_info(data, contours)


def main():
    fig, ax = plt.subplots(figsize=fig_size, dpi=50)

    ax.invert_yaxis()
    ax.set_axis_off()

    contours = open_contours(contours_file)

    if len(contours) == 0:
        return

    # plot_contours(contours) # for testing input data
    data = open_data(file_name)

    if len(data) == 0:
        return

    plot_pandasData(data, ax)
    data_and_contours = connect_points_with_contours(data, contours)

    if len(data_and_contours) == 0:
        return

    grouped_list_data = grouped_data_by_type(data_and_contours)
    plot_listData(grouped_list_data, contours, ax)

    # plt.show() # for testing input data

    clusters = clustering(data, contours)
    plot_clusters(data, clusters, ax)

    # To show the graphics:
    # plt.legend()
    # plt.title(cell_name + fl_all)
    # plt.show()

    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0, frameon=False, dpi=300, format='pdf')



if __name__ == '__main__':
    main()
    # info_for_all_neighbors()
