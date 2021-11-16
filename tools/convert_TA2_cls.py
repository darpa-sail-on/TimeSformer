import json
import os
import pandas as pd
import cv2
from tqdm import trange
import math
import pdb


def get_n_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_length = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        frame_length += 1
    cap.release()

    return frame_length

def load_test_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, header=None)
    labels = {'action':[],'perspective':[],'location':[],'type':[],'relation':[]}
    exist_label = {'action':[],'perspective':[],'location':[],'type':[],'relation':[]}
    print('parsing testing labels...')
    perspective_dict = {'Front': 'Front', 'Top': 'Top', 'Side': 'Side', 'Back': 'Back', 'Bottom': 'Bottom', 'Sides': 'Side', 'Bottoom': 'Bottom', 'Behind': 'Back', 'Font': 'Front'}
    for i in range(1, data.shape[0]):
        # action label
        if data.iloc[i,4].capitalize() not in exist_label['action']:
            class_name = data.iloc[i,4].capitalize()
            labels['action'].append(class_name) # class name
            exist_label['action'].append(class_name) # class index

        # perspective label
        class_name = perspective_dict[data.iloc[i,6].capitalize()]
        if class_name not in exist_label['perspective']:
            labels['perspective'].append(class_name) # class name
            exist_label['perspective'].append(class_name) # class index
        # location label
        if data.iloc[i,7].capitalize() not in exist_label['location']:
            class_name = data.iloc[i,7].capitalize()
            labels['location'].append(class_name) # class name
            exist_label['location'].append(class_name) # class index
        # type label
        if data.iloc[i,8].capitalize() not in exist_label['type']:
            class_name = data.iloc[i,8].capitalize()
            labels['type'].append(class_name) # class name
            exist_label['type'].append(class_name) # class index
        if data.iloc[i,10] not in exist_label['type']:
            data.iloc[i,10] = str(data.iloc[i,10])
            if data.iloc[i,10] != 'nan' and data.iloc[i,10].capitalize() not in exist_label['type']:
                class_name = data.iloc[i,10].capitalize()
                labels['type'].append(class_name) # class name
                exist_label['type'].append(class_name) # class index

        # relation label
        class_name = data.iloc[i,9].capitalize()
        if class_name not in exist_label['relation']:
            labels['relation'].append(class_name) # class name
            exist_label['relation'].append(class_name) # class index
        class_name = data.iloc[i,11].capitalize()
        if class_name not in exist_label['relation']:
            if class_name != 'None':
                labels['relation'].append(class_name) # class name
                exist_label['relation'].append(class_name) # class index

    print('testing classes: {}'.format(len(labels['action'])))
    print('perspective: {}'.format(len(labels['perspective'])))
    print('location: {}'.format(len(labels['location'])))
    print('type: {}'.format(len(labels['type'])))
    print('relation: {}'.format(len(labels['relation'])))

    return labels


def load_train_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, header=None)
    labels = {'action':[],'perspective':[],'location':[],'type':[],'relation':[]}
    exist_label = {'action':[],'perspective':[],'location':[],'type':[],'relation':[]}
    print('parsing known labels...')
    for i in range(1, data.shape[0]):
        # action label
        if data.iloc[i,3].capitalize() not in exist_label['action']:
            class_name = data.iloc[i,3].capitalize()
            labels['action'].append(class_name) # class name
            exist_label['action'].append(class_name) # class index

        # perspective label
        if data.iloc[i,4].capitalize() not in exist_label['perspective']:
            class_name = data.iloc[i,4].capitalize()
            labels['perspective'].append(class_name) # class name
            exist_label['perspective'].append(class_name) # class index
        # location label
        if data.iloc[i,5].capitalize() not in exist_label['location']:
            class_name = data.iloc[i,5].capitalize()
            labels['location'].append(class_name) # class name
            exist_label['location'].append(class_name) # class index
        # type label
        if data.iloc[i,6].capitalize() not in exist_label['type']:
            class_name = data.iloc[i,6].capitalize()
            labels['type'].append(class_name) # class name
            exist_label['type'].append(class_name) # class index
        if data.iloc[i,8] not in exist_label['type']:
            if not math.isnan(data.iloc[i,8]) and data.iloc[i,8].capitalize() not in exist_label['type']:
                class_name = data.iloc[i,8].capitalize()
                labels['type'].append(class_name) # class name
                exist_label['type'].append(class_name) # class index
        # relation label
        if data.iloc[i,7].capitalize() not in exist_label['relation']:
            class_name = data.iloc[i,7].capitalize()
            labels['relation'].append(class_name) # class name
            exist_label['relation'].append(class_name) # class index
        if data.iloc[i,9].capitalize() not in exist_label['relation']:
            if data.iloc[i,9] != 'None':
                class_name = data.iloc[i,9].capitalize()
                labels['relation'].append(class_name) # class name
                exist_label['relation'].append(class_name) # class index

    print('known classes: {}'.format(len(labels['action'])))
    print('perspective: {}'.format(len(labels['perspective'])))
    print('location: {}'.format(len(labels['location'])))
    print('type: {}'.format(len(labels['type'])))
    print('relation: {}'.format(len(labels['relation'])))

    return labels

def convert_train_to_dict(dataset_path, csv_path):
    data = pd.read_csv(csv_path, header=None)
    train_keys, val_keys = [], []
    train_key_labels = {'action':[],'perspective':[],'location':[],'type1':[],'relation1':[],'type2':[],'relation2':[]}
    val_key_labels = {'action':[],'perspective':[],'location':[],'type1':[],'relation1':[],'type2':[],'relation2':[]}

    for i in range(1, data.shape[0]):
        video_name = data.iloc[i, 0] # video name
        action_name = data.iloc[i, 3].capitalize() # action name
        perspective_name = data.iloc[i, 4].capitalize() # perspective name
        location_name = data.iloc[i, 5].capitalize() # location name
        type_name1 = data.iloc[i, 6].capitalize() # type name
        try:
            type_name2 = data.iloc[i, 8].capitalize() # type name
        except:
            type_name2 = 'None'
        relation_name1 = data.iloc[i, 7].capitalize() # relation name
        relation_name2 = data.iloc[i, 9].capitalize() # relation name

        video_path = os.path.join(dataset_path, video_name)
        if i%10 != 0: # 90% training
            train_keys.append(video_path)
            train_key_labels['action'].append(action_name)
            train_key_labels['perspective'].append(perspective_name)
            train_key_labels['location'].append(location_name)
            train_key_labels['type1'].append(type_name1)
            train_key_labels['type2'].append(type_name2)
            train_key_labels['relation1'].append(relation_name1)
            train_key_labels['relation2'].append(relation_name2)

        else: # 10% validation
            val_keys.append(video_path)
            val_key_labels['action'].append(action_name)
            val_key_labels['perspective'].append(perspective_name)
            val_key_labels['location'].append(location_name)
            val_key_labels['type1'].append(type_name1)
            val_key_labels['type2'].append(type_name2)
            val_key_labels['relation1'].append(relation_name1)
            val_key_labels['relation2'].append(relation_name2)

    return train_keys, val_keys, train_key_labels, val_key_labels

def convert_trainval_to_json(train_csv_path, video_dir_path):
    dst_data = {}
    dst_data['database'] = {}
    train_database, val_database = {}, {}
    labels = load_train_labels(train_csv_path)
    train_keys, val_keys, train_key_labels, val_key_labels = convert_train_to_dict(video_dir_path, train_csv_path)
    dst_data['labels'] = labels['action']
    dst_data['perspective'] = labels['perspective']
    dst_data['location'] = labels['location']
    dst_data['type'] = labels['type']
    dst_data['relation'] = labels['relation']

    # save database
    for i in range(len(train_keys)):
        key = train_keys[i] # video name
        train_database[key] = {}
        train_database[key]['subset'] = 'training'
        label = train_key_labels['action'][i]
        perspective = train_key_labels['perspective'][i]
        location = train_key_labels['location'][i]
        type1 = train_key_labels['type1'][i]
        relation1 = train_key_labels['relation1'][i]
        type2 = train_key_labels['type2'][i]
        relation2 = train_key_labels['relation2'][i]
        train_database[key]['annotations'] = {'label': label, 'perspective': perspective, 'location': location, 'type1': type1, 'relation1': relation1,
                                              'type2': type2, 'relation2': relation2}

    for i in range(len(val_keys)):
        key = val_keys[i] # video name
        val_database[key] = {}
        val_database[key]['subset'] = 'validation'
        label = val_key_labels['action'][i]
        perspective = val_key_labels['perspective'][i]
        location = val_key_labels['location'][i]
        type1 = val_key_labels['type1'][i]
        relation1 = val_key_labels['relation1'][i]
        type2 = val_key_labels['type2'][i]
        relation2 = val_key_labels['relation2'][i]
        val_database[key]['annotations'] = {'label': label, 'perspective': perspective, 'location': location, 'type1': type1, 'relation1': relation1,
                                            'type2': type2, 'relation2': relation2}

    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)

    return dst_data, labels

def convert_test_to_dict(labels, dataset_path, csv_path):
    #labels = load_test_labels(csv_path)
    #pdb.set_trace()
    data = pd.read_csv(csv_path, header=None)
    test_keys = []
    test_key_labels = {'action':[],'perspective':[],'location':[],'type1':[],'relation1':[],'type2':[],'relation2':[]}
    unknown_keys = []
    unknown_key_labels = {'action':[],'perspective':[],'location':[],'type1':[],'relation1':[],'type2':[],'relation2':[]}

    perspective_dict = {'Front': 'Front', 'Top': 'Top', 'Side': 'Side', 'Back': 'Back', 'Bottom': 'Bottom', 'Sides': 'Side', 'Bottoom': 'Bottom', 'Behind': 'Back', 'Font': 'Front'}

    for i in range(1, data.shape[0]):
        video_name = data.iloc[i, 0] # video name
        action_name = data.iloc[i, 4].capitalize() # action name
        perspective_name = perspective_dict[data.iloc[i, 6].capitalize()] # perspective name
        location_name = data.iloc[i, 7].capitalize() # location name
        type_name1 = data.iloc[i, 8].capitalize() # type name
        try:
            type_name2 = data.iloc[i, 10].capitalize() # type name
        except:
            type_name2 = 'None'
        relation_name1 = data.iloc[i, 9].capitalize() # relation name
        relation_name2 = data.iloc[i, 11].capitalize() # relation name
        video_path = os.path.join(dataset_path, video_name)
        # only collect testing videos with known classes
        flag = False
        if action_name in labels['action'] and relation_name1 in labels['relation']:
            if relation_name2 == 'None':
                flag = True
            else:
                if relation_name2 in labels['relation']:
                    flag = True

        if flag:  
            test_keys.append(video_path)
            test_key_labels['action'].append(action_name)
            test_key_labels['perspective'].append(perspective_name)
            test_key_labels['location'].append(location_name)
            test_key_labels['type1'].append(type_name1)
            test_key_labels['type2'].append(type_name2)
            test_key_labels['relation1'].append(relation_name1)
            test_key_labels['relation2'].append(relation_name2)
        else:
            unknown_keys.append(video_path)
            unknown_key_labels['action'].append(action_name)
            unknown_key_labels['perspective'].append(perspective_name)
            unknown_key_labels['location'].append(location_name)
            unknown_key_labels['type1'].append(type_name1)
            unknown_key_labels['type2'].append(type_name2)
            unknown_key_labels['relation1'].append(relation_name1)
            unknown_key_labels['relation2'].append(relation_name2)

    return test_keys, test_key_labels, unknown_keys, unknown_key_labels


def convert_test_to_json(dst_data, labels, test_csv_path, video_dir_path, dst_json_path):
    test_database = {}
    test_keys, test_key_labels, _, _ = convert_test_to_dict(labels, video_dir_path, test_csv_path)

    # save database
    for i in range(len(test_keys)):
        key = test_keys[i]
        test_database[key] = {}
        test_database[key]['subset'] = 'test'
        label = test_key_labels['action'][i]
        perspective = test_key_labels['perspective'][i]
        location = test_key_labels['location'][i]
        type1 = test_key_labels['type1'][i]
        relation1 = test_key_labels['relation1'][i]
        type2 = test_key_labels['type2'][i]
        relation2 = test_key_labels['relation2'][i]
        test_database[key]['annotations'] = {'label': label, 'perspective': perspective, 'location': location, 'type1': type1, 'relation1': relation1,
                                             'type2': type2, 'relation2': relation2}

    dst_data['database'].update(test_database)
    num = len(dst_data['database'].items())

    cnt = 0
    for video_path, frame_range in dst_data['database'].items():
        cnt += 1
        if cnt % 1000 == 0:
            print("parsing sample {}/{}...".format(cnt, num))
        n_frames = get_n_frames(video_path)
        frame_range['annotations']['segment'] = (1, n_frames)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)

def convert_TA2_to_evm(train_csv_path, train_video_path, test_csv_path, test_video_path, dst_json_path):
    dst_data = {}
    dst_data['database'] = {}
    train_database, known_database, unknown_database = {}, {}, {}

    # parse known samples
    train_labels = load_train_labels(train_csv_path)
    train_keys, known_keys, train_key_labels, known_key_labels = convert_train_to_dict(train_video_path, train_csv_path)
    dst_data['labels'] = train_labels['action']
    dst_data['perspective'] = train_labels['perspective']
    dst_data['location'] = train_labels['location']
    dst_data['type'] = train_labels['type']
    dst_data['relation'] = train_labels['relation']

    for i in range(len(train_keys)):
        key = train_keys[i] # video name
        train_database[key] = {}
        train_database[key]['subset'] = 'training'
        label = train_key_labels['action'][i]
        perspective = train_key_labels['perspective'][i]
        location = train_key_labels['location'][i]
        type1 = train_key_labels['type1'][i]
        relation1 = train_key_labels['relation1'][i]
        type2 = train_key_labels['type2'][i]
        relation2 = train_key_labels['relation2'][i]
        train_database[key]['annotations'] = {'label': label, 'perspective': perspective, 'location': location, 'type1': type1, 'relation1': relation1,
                                              'type2': type2, 'relation2': relation2}

    for i in range(len(known_keys)):
        key = known_keys[i]
        train_database[key] = {}
        train_database[key]['subset'] = 'training'
        label = known_key_labels['action'][i]
        perspective = known_key_labels['perspective'][i]
        location = known_key_labels['location'][i]
        type1 = known_key_labels['type1'][i]
        relation1 = known_key_labels['relation1'][i]
        type2 = known_key_labels['type2'][i]
        relation2 = known_key_labels['relation2'][i]
        train_database[key]['annotations'] = {'label': label, 'perspective': perspective, 'location': location, 'type1': type1, 'relation1': relation1,
                                              'type2': type2, 'relation2': relation2}
    
    # parse unknown samples
    test_labels = load_test_labels(test_csv_path)
    # update the class set
    dst_data['labels'] = dst_data['labels'] + list(set(test_labels['action']).difference(set(train_labels['action'])))
    dst_data['relation'] = dst_data['relation'] + list(set(test_labels['relation']).difference(set(train_labels['relation'])))

    known_keys, known_key_labels, unknown_keys, unknown_key_labels = convert_test_to_dict(train_labels, test_video_path, test_csv_path)

    for i in range(len(known_keys)):
        key = known_keys[i]
        known_database[key] = {}
        known_database[key]['subset'] = 'known'
        label = known_key_labels['action'][i]
        perspective = known_key_labels['perspective'][i]
        location = known_key_labels['location'][i]
        type1 = known_key_labels['type1'][i]
        relation1 = known_key_labels['relation1'][i]
        type2 = known_key_labels['type2'][i]
        relation2 = known_key_labels['relation2'][i]
        known_database[key]['annotations'] = {'label': label, 'perspective': perspective, 'location': location, 'type1': type1, 'relation1': relation1,
                                              'type2': type2, 'relation2': relation2}
    for i in range(len(unknown_keys)):
        key = unknown_keys[i]
        unknown_database[key] = {}
        unknown_database[key]['subset'] = 'unknown'
        label = unknown_key_labels['action'][i]
        perspective = unknown_key_labels['perspective'][i]
        location = unknown_key_labels['location'][i]
        type1 = unknown_key_labels['type1'][i]
        relation1 = unknown_key_labels['relation1'][i]
        type2 = unknown_key_labels['type2'][i]
        relation2 = unknown_key_labels['relation2'][i]
        unknown_database[key]['annotations'] = {'label': label, 'perspective': perspective, 'location': location, 'type1': type1, 'relation1': relation1,
                                              'type2': type2, 'relation2': relation2}

    dst_data['database'].update(train_database)
    dst_data['database'].update(known_database)
    dst_data['database'].update(unknown_database)

    # get the number of frames for each video
    cnt = 0
    for video_path, frame_range in dst_data['database'].items():
        cnt += 1
        if cnt % 1000 == 0:
            print("parsing sample {}/{}...".format(cnt, len(train_database)+len(known_database)+len(unknown_database)))
        n_frames = get_n_frames(video_path)
        frame_range['annotations']['segment'] = (1, n_frames)
    # save json files
    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    dir_path = '/data/datasets/m24-activity/' # Path of label directory
    dst_path = '/data/datasets/m24-activity/' # Directory path of dst json file.
    dst_json_path = dst_path + 'm24_activity_cls.json'

    train_video_path = dir_path + 'all_vids/' # Path of train video directory
    train_csv_path = dir_path + 'training_m24.csv'
    dst_data, train_labels = convert_trainval_to_json(train_csv_path, train_video_path)

    test_video_path = dir_path + '/updated-dataset/Classes/' # Path of test video directory
    test_csv_path = dir_path + 'internal_eval2.csv' # need to fix the typo at line 909
    convert_test_to_json(dst_data, train_labels, test_csv_path, test_video_path, dst_json_path)

    dst_json_path = dst_path + 'm24_activity_evm.json'
    convert_TA2_to_evm(train_csv_path, train_video_path, test_csv_path, test_video_path, dst_json_path) # evm training and evaluation



