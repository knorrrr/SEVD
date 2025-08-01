import os
import carla
import numpy as np
import pygame
import cv2
import traceback
import json
from bounding_boxes import create_kitti_datapoint
import concurrent.futures


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera, h, w, fov):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """
        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(
            vehicle, camera, h, w, fov) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def get_bounding_box(vehicle, camera, h, w, fov):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(
            bb_cords, vehicle, camera)[:3, :]

        cords_y_minus_z_x = np.concatenate(
            [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])

        calibration = np.identity(3)
        calibration[0, 2] = w / 2.0
        calibration[1, 2] = h / 2.0
        calibration[0, 0] = calibration[1, 1] = w / \
            (2.0 * np.tan(fov * np.pi / 360.0))

        bbox = np.transpose(np.dot(calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate(
            [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(
            world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(
            vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(
            sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    @staticmethod
    def get_bounding_boxes_parked_vehicles(bboxes, camera, h, w, fov):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """
        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box_parked_vehicle(
            vehicle, camera, h, w, fov) for vehicle in bboxes]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def get_bounding_box_parked_vehicle(bbox, camera, h, w, fov):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """
        bb_cords = ClientSideBoundingBoxes._bounding_box_to_world(bbox)
        cords_x_y_z = ClientSideBoundingBoxes._world_to_sensor(bb_cords, camera)[
            :3, :]
        cords_y_minus_z_x = np.concatenate(
            [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        calibration = np.identity(3)
        calibration[0, 2] = w / 2.0
        calibration[1, 2] = h / 2.0
        calibration[0, 0] = calibration[1, 1] = w / \
            (2.0 * np.tan(fov * np.pi / 360.0))
        bbox = np.transpose(np.dot(calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate(
            [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _bounding_box_to_world(bbox):
        extent = bbox.extent
        cords = np.zeros((8, 4))
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])

        world_matrix = ClientSideBoundingBoxes.get_matrix(bbox)

        world_cords = np.dot(world_matrix, np.transpose(cords))

        return world_cords

    @staticmethod
    def _create_bb_points_parked(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        if isinstance(vehicle, carla.BoundingBox):
            extent = vehicle.extent
        else:
            extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords


edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
         [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]


def saveAllSensors(out_root_folder, sensor_datas, sensor_types, world):
    sensor_datas.pop(0)

    lidar_data = {}
    rgb_camera = {}
    depth_camera = {}
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(len(sensor_datas)):
            try:
                (sensor_data, sensor, vehicle) = sensor_datas[i]
            except:
                try:
                    (sensor_data, sensor) = sensor_datas[i]
                    vehicle = None
                except Exception as error:
                    print("An exception occurred in saveAllSensors:", error)
                    traceback.print_exc()
            sensor_name = sensor_types[i]

            if (sensor_name.find('dvs') != -1):
                try:
                    lidar = sensor_name.replace("dvs_camera","lidar")

                    future = executor.submit(saveLidars, sensor_data, os.path.join(out_root_folder, lidar), 
                                                                       lidar_data["lidar-front"])
                    futures.append(future)
                except Exception as error:
                    print("An exception occurred in lidar sensor find:", error)
                    traceback.print_exc()

            if (sensor_name.find('lidar') != -1):
                lidar_data[sensor_name] = sensor_data


            if (sensor_name.find('optical_flow') != -1):
                optical_camera_callback(
                    sensor_data, os.path.join(out_root_folder, sensor_name))

            if (sensor_name.find('instance_segmentation_camera') != -1):
                saveISImage(sensor_data, os.path.join(
                    out_root_folder, sensor_name))
                pass

            if (sensor_name.find('semantic_segmentation_camera') != -1):
                saveSegImage(sensor_data, os.path.join(
                    out_root_folder, sensor_name))

            if (sensor_name.find('depth_camera') != -1):
                depth_camera[sensor_name] = sensor_data
                saveDepthImage(sensor_data, os.path.join(
                    out_root_folder, sensor_name))

            if (sensor_name.find('rgb_camera') != -1):
                saveOnlyRgb(sensor_data, os.path.join(
                    out_root_folder, sensor_name))
                # try:
                #     rgb_camera[sensor_name] = (
                #         sensor_data[i], os.path.join(out_root_folder, sensor_name))
                #     dvs = sensor_name.replace("rgb", "dvs")
                #     print("DVS Camera: ", dvs)
                #     depth = sensor_name.replace("rgb", "depth")
                #     rgb_file_path = os.path.join(
                #         out_root_folder, sensor_name)
                #     future = executor.submit(saveRgbImage, sensor_data, rgb_file_path,
                #                              world, sensor, vehicle, dvs_camera[dvs], depth_camera[depth])
                #     futures.append(future)
                # except Exception as error:
                #     print("An exception occurred in rgb_camera sensor find:", error)
                #     traceback.print_exc()

            if (sensor_name.find('imu') != -1):
                saveImu(sensor_data, os.path.join(
                    out_root_folder, sensor_name), sensor_name)

            if (sensor_name.find('gnss') != -1):
                saveGnss(sensor_data, os.path.join(
                    out_root_folder, sensor_name), sensor_name)

        concurrent.futures.wait(
            futures, return_when=concurrent.futures.ALL_COMPLETED)
    return


def saveSnapshot(output, filepath):
    return


def saveSteeringAngle(value, filepath):
    with open(filepath + "/steering_norm.txt", 'a') as fp:
        fp.writelines(str(value) + ", ")
    with open(filepath + "/steering_true.txt", 'a') as fp:
        fp.writelines(str(70*value) + ", ")


def saveGnss(output, filepath, sensor_name):
    with open(filepath + "/" + sensor_name + ".txt", 'a') as fp:
        fp.writelines(str(output) + ", ")
        fp.writelines(str(output.transform) + "\n")


def saveImu(output, filepath, sensor_name):
    with open(filepath + "/" + sensor_name + ".txt", 'a') as fp:
        fp.writelines(str(output) + ", ")
        fp.writelines(str(output.transform) + "\n")


def saveLidar(output, filepath):
    output.save_to_disk(filepath + '/%05d' % output.frame)
    with open(filepath + "/lidar_metadata.txt", 'a') as fp:
        fp.writelines(str(output) + ", ")
        fp.writelines(str(output.transform) + "\n")


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]


def get_2d_bounding_box(points):
    sorted_points = sort_points_clockwise(points)

    min_x = min(point[0] for point in sorted_points)
    min_y = min(point[1] for point in sorted_points)
    max_x = max(point[0] for point in sorted_points)
    max_y = max(point[1] for point in sorted_points)

    return int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)


def get_bounding_box_center(bounding_box):
    x, y, w, h = bounding_box
    center_x = x + w // 2
    center_y = y + h // 2
    return center_x, center_y


def sort_points_clockwise(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]


def draw_bounding_box(image, bounding_box, color=(0, 255, 0), thickness=2):
    x, y, w, h = bounding_box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


def draw_bounding_box_center(image, center, color=(0, 0, 0), radius=5):
    cv2.circle(image, center, radius, color, -1)


def draw_bounding_box_corners(image, points, color=(0, 0, 255), thickness=2):
    for i in range(4):
        cv2.line(image, tuple(points[i]), tuple(
            points[(i + 1) % 4]), color, thickness)


def save_pascal_voc_format(bounding_boxes_with_ids, file_path, image_filename, image_w, image_h):
    with open(file_path, 'w') as file:
        file.write(f'<?xml version="1.0" encoding="UTF-8"?>\n')
        file.write(f'<annotation>\n')
        file.write(f'    <filename>{image_filename}</filename>\n')
        file.write(f'    <size>\n')
        file.write(f'        <width>{image_w}</width>\n')
        file.write(f'        <height>{image_h}</height>\n')
        file.write(f'        <depth>3</depth>\n')
        file.write(f'    </size>\n')

        for obj_id, class_name, bbox in bounding_boxes_with_ids:
            xmin, ymin, width, height = bbox
            file.write(f'    <object>\n')
            file.write(f'        <name>{class_name}</name>\n')
            file.write(f'        <object_id>{obj_id}</object_id>\n')
            file.write(f'        <bndbox>\n')
            file.write(f'            <xmin>{xmin}</xmin>\n')
            file.write(f'            <ymin>{ymin}</ymin>\n')
            file.write(f'            <xmax>{xmin + width}</xmax>\n')
            file.write(f'            <ymax>{ymin + height}</ymax>\n')
            file.write(f'        </bndbox>\n')
            file.write(f'    </object>\n')

        file.write(f'</annotation>\n')


def save_coco_format(bounding_boxes, file_path, id, image_filename, image_w, image_h):
    coco = {
        "car": 1,
        "truck": 2,
        "van": 3,
        "pedestrian": 4,
        "motorcycle": 5,
        "bicycle": 6,
    }

    coco_data = {
        "images": [
            {
                "id": id,
                "file_name": image_filename,
                "width": image_w,
                "height": image_h,
            }
        ],
        "annotations": [
        ],
        "categories": [
            {"id": 1, "name": "car", "supercategory": "vehicle"},
            {"id": 2, "name": "truck", "supercategory": "vehicle"},
            {"id": 3, "name": "van", "supercategory": "vehicle"},
            {"id": 4, "name": "pedestrian", "supercategory": "human"},
            {"id": 5, "name": "motorcycle", "supercategory": "vehicle"},
            {"id": 6, "name": "bicycle", "supercategory": "vehicle"}
        ]
    }
    for obj_id, class_name, bbox in bounding_boxes:
        coco_data["annotations"].append({
            "id": obj_id,
            "image_id": id,
            "category_id": coco[class_name],
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "segmentation": [],
        })
    with open(file_path, 'w') as file:
        json.dump(coco_data, file, indent=4)


def save_kitti_3d_format(annotations, filepath):
    with open(filepath, "w") as file:
        for element in annotations:
            file.write(str(element) + "\n")

def saveOnlyRgb(output, filepath):
    img = np.frombuffer(output.raw_data, dtype=np.uint8).reshape(
            (output.height, output.width, 4))
    output_file = os.path.join(
            filepath, f'{output.frame}.png')
    cv2.imwrite(output_file, img)

def saveLidars(dvs, filepath, output):
    # Save the lidar data to disk
    output.save_to_disk(filepath + '/%05d' % output.frame)
    with open(filepath + "/lidar_metadata.txt", 'a') as fp:
        fp.writelines(str(output) + ", ")
        fp.writelines(str(output.transform) + "\n")
    # Save the DVS data if available
    try:
        dvs_events = np.frombuffer(dvs.raw_data, dtype=np.dtype([
            ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)
        ]))
        output_file_path = os.path.join(
            filepath.replace("lidar", "dvs_camera"), f'dvs-{output.frame}.npz')
        np.savez_compressed(output_file_path, dvs_events=dvs_events)
    except Exception as error:
        print("An exception occurred while saving DVS data:", error)

def saveRgbImage(output, filepath, world, sensor, ego_vehicle, dvs, depth):
    try:
        dvs_events = np.frombuffer(dvs.raw_data, dtype=np.dtype([
            ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)
        ]))
        output_file_path = os.path.join(
            filepath, f'dvs-{output.frame}-xytp.npz')
        np.savez_compressed(output_file_path, dvs_events=dvs_events)
        dvs_events2 = np.frombuffer(dvs.raw_data, dtype=np.dtype([
            ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
        dvs_img = np.zeros((dvs.height, dvs.width, 3), dtype=np.uint8)
        dvs_img[dvs_events2[:]['y'], dvs_events2[:]
                ['x'], dvs_events2[:]['pol'] * 2] = 255
        surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))

        array = np.frombuffer(depth.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (depth.height, depth.width, 4))
        array = array[:, :, :3]
        array = array.astype(np.float32)
        normalized_depth = np.dot(array, [65536.0, 256.0, 1.0])
        normalized_depth /= 16777215.0
        deptharray = normalized_depth * 1000

        img = np.frombuffer(output.raw_data, dtype=np.uint8).reshape(
            (output.height, output.width, 4))

        # All labels in CityObjectLabel
        # ['Any', 'Bicycle', 'Bridge', 'Buildings', 'Bus', 'Car', 'Dynamic', 'Fences', 'Ground', 'GuardRail', 'Motorcycle', 'NONE', 'Other', 'Pedestrians', 'Poles', 'RailTrack', 'Rider', 'RoadLines', 'Roads', 'Sidewalks', 'Sky', 'Static', 'Terrain', 'TrafficLight', 'TrafficSigns', 'Train', 'Truck', 'Vegetation', 'Walls', 'Water', '__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__module__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'name', 'names', 'numerator', 'real', 'to_bytes', 'values']

        dvsbb = []
        rgbbb = []

        calibration = np.identity(3)
        calibration[0, 2] = output.width / 2.0
        calibration[1, 2] = output.height / 2.0
        calibration[0, 0] = calibration[1, 1] = output.width / \
            (2.0 * np.tan(output.fov * np.pi / 360.0))
        kitti3dbb = []
        kitti3dbbDVS = []

        for vehicle in world.get_actors().filter("*vehicle*"):
            bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(
                [vehicle], sensor, output.height, output.width, output.fov)
            for bbox in bounding_boxes:
                points = [(int(bbox[i, 0]), int(bbox[i, 1]))
                          for i in range(8)]
                bounding_box = get_2d_bounding_box(
                    np.array(points, dtype=np.int32))
                min_x, min_y, xdiff, ydiff = bounding_box
                isDvs = is_dvs_event_inside_bbox(
                    dvs_events, min_x, min_y, min_x + xdiff, min_y + ydiff)
                transform = output.transform
                image, datapoint, camera_bbox = create_kitti_datapoint(
                    vehicle, sensor, calibration, img, deptharray, transform, bbox)
                if datapoint is not None:
                    kitti3dbb.append(datapoint)
                    rgbbb.append((vehicle.id, vehicle.attributes.get(
                        'base_type'), (min_x, min_y, xdiff, ydiff)))
                    if isDvs == True:
                        kitti3dbbDVS.append(datapoint)
                        dvsbb.append((vehicle.id, vehicle.attributes.get(
                            'base_type'), (min_x, min_y, xdiff, ydiff)))

        for vehicle in world.get_actors().filter("*pedestrian*"):
            bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(
                [vehicle], sensor, output.height, output.width, output.fov)
            for bbox in bounding_boxes:
                points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
                bounding_box = get_2d_bounding_box(
                    np.array(points, dtype=np.int32))
                min_x, min_y, xdiff, ydiff = bounding_box
                transform = output.transform
                image, datapoint, camera_bbox = create_kitti_datapoint(
                    vehicle, sensor, calibration, img, deptharray, transform, bbox)
                isDvs = is_dvs_event_inside_bbox(
                    dvs_events, min_x, min_y, min_x + xdiff, min_y + ydiff)
                if datapoint is not None:
                    kitti3dbb.append(datapoint)
                    rgbbb.append((vehicle.id, 'pedestrian',
                                 (min_x, min_y, xdiff, ydiff)))
                    if isDvs == True:
                        kitti3dbbDVS.append(datapoint)
                        dvsbb.append((vehicle.id, 'pedestrian',
                                     (min_x, min_y, xdiff, ydiff)))

        output_file = os.path.join(
            filepath, f'{output.frame}.png')
        cv2.imwrite(output_file, img)

        output_file = os.path.join(filepath, f'dvs-{output.frame}.png')
        pygame.image.save(surface, output_file)

        # save_pascal_voc_format(rgbbb, os.path.join(
        #     filepath, f'{output.frame}.xml'), f'{output.frame}.png', output.width, output.height)
        # save_coco_format(rgbbb, os.path.join(
        #     filepath, f'{output.frame}.json'), output.frame, f'{output.frame}.png', output.width, output.height)

        # save_pascal_voc_format(dvsbb, os.path.join(
        #     filepath, f'dvs-{output.frame}.xml'), f'dvs-{output.frame}.png', output.width, output.height)
        # save_coco_format(dvsbb, os.path.join(
        #     filepath, f'dvs-{output.frame}.json'), output.frame, f'dvs-{output.frame}.png', output.width, output.height)

        # save_kitti_3d_format(kitti3dbb, os.path.join(
        #     filepath, f'{output.frame}.txt'))
        # save_kitti_3d_format(kitti3dbbDVS, os.path.join(
        #     filepath, f'dvs-{output.frame}.txt'))

    except Exception as error:
        print("An exception occurred:", error)
        traceback.print_exc()


def saveISImage(output, filepath):
    try:
        output.save_to_disk(filepath + '/%05d' % output.frame)
        with open(filepath + "/rgb_camera_metadata.txt", 'a') as fp:
            fp.writelines(str(output) + ", ")
            fp.writelines(str(output.transform) + "\n")
    except Exception as error:
        print("An exception occurred:", error)


def is_dvs_event_inside_bbox(event, x_min, y_min, x_max, y_max):
    # Extract x, y, and polarity
    x, y, polarity = event['x'], event['y'], event['pol']
    is_inside_bbox = np.logical_and(
        np.logical_and(x_min <= event['x'], event['x'] <= x_max),
        np.logical_and(y_min <= event['y'], event['y'] <= y_max)
    )

    if is_inside_bbox.any():
        return True
    return False


def dvs_callback(data, filepath):
    timestamp = data.timestamp
    dvs_events = np.frombuffer(data.raw_data, dtype=np.dtype([
        ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
    dvs_img = np.zeros((data.height, data.width, 3), dtype=np.uint8)
    dvs_img[dvs_events[:]['y'], dvs_events[:]
            ['x'], dvs_events[:]['pol'] * 2] = 255
    surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
    output_file = os.path.join(filepath, f'{data.frame}.png')
    pygame.image.save(surface, output_file)


def optical_camera_callback(image, filepath):
    image_data = np.frombuffer(image.raw_data, dtype=np.float32)
    image_data = image_data.reshape((image.height, image.width, 2))
    filename = os.path.join(filepath, f"{image.frame}.npz")
    data_dict = {'flow': image_data}
    np.savez_compressed(filename, **data_dict)

    data = image.get_color_coded_flow()
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (data.height, data.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    filename = os.path.join(filepath, f"{image.frame}.png")
    cv2.imwrite(filename, array)


def get_intrinsic_matrix(height, width, fov):

    width = int(width)
    height = int(height)
    fov = float(fov)

    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[0, 0] = k[1, 1] = width / (2.0 * np.tan(fov * np.pi / 360.0))

    return k


def save_calibration_matrices(filename, intrinsic_mat):
    ravel_mode = 'C'
    P0 = intrinsic_mat
    P0 = np.column_stack((P0, np.array([0, 0, 0])))
    P0 = np.ravel(P0, order=ravel_mode)

    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten(ravel_mode).squeeze()))))

    # All matrices are written on a line with spacing
    with open(filename, 'w') as f:
        for i in range(4):  # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), P0)


def saveDepthImage(output, filepath):
    output.convert(carla.ColorConverter.Depth)
    output.save_to_disk(filepath + '/%05d' % output.frame)
    with open(filepath + "/depth_camera_metadata.txt", 'a') as fp:
        fp.writelines(str(output) + ", ")
        fp.writelines(str(output.transform) + "\n")


def saveSegImage(output, filepath):
    output.convert(carla.ColorConverter.CityScapesPalette)
    output.save_to_disk(filepath + '/%05d' % output.frame)

    image_data = np.array(output.raw_data)
    image_data = image_data.reshape((output.height, output.width, 4))
    semantic_image = image_data[:, :, :3]

    with open(filepath + "/seg_camera_metadata.txt", 'a') as fp:
        fp.writelines(str(output) + ", ")
        fp.writelines(str(output.transform) + "\n")


def saveDvsImage(output, filepath):
    output.convert(carla.ColorConverter.CityScapesPalette)
    output.save_to_disk(filepath + '/%05d' % output.frame)
    with open(filepath + "/seg_camera_metadata.txt", 'a') as fp:
        fp.writelines(str(output) + ", ")
        fp.writelines(str(output.transform) + "\n")
