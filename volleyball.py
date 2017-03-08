import numpy as np

TRAIN_SEQS = [1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54]
VAL_SEQS = [0,2,8,12,17,19,24,26,27,28,30,33,46,49,51]
TEST_SEQS = [4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47]

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

NUM_ACTIVITIES = 8

ACTIONS = ['blocking','digging','falling','jumping',
           'moving','setting','spiking','standing',
           'waiting']
NUM_ACTIONS = 9

def volley_read_annotations(path):
  """reading annotations for the given sequence"""
  annotations = {}

  gact_to_id = { name : i for i, name in enumerate(ACTIVITIES) }
  act_to_id = { name : i for i, name in enumerate(ACTIONS) }

  with open(path) as f:
    for l in f.readlines():
      values = l[:-1].split(' ')
      file_name = values[0]
      activity = gact_to_id[values[1]]

      values = values[2:]
      num_people = len(values) // 5

      action_names = values[4::5]
      actions = [act_to_id[name]
                 for name in action_names]

      def _read_bbox(xywh):
        x,y,w,h = map(int, xywh)
        return y,x,y+h,x+w
      bboxes = np.array([_read_bbox(values[i:i+4])
                        for i in range(0, 5*num_people, 5)])

      fid = int(file_name.split('.')[0])
      annotations[fid] = {
        'file_name' : file_name,
        'group_activity' : activity,
        'actions': actions,
        'bboxes' : bboxes,
      }
  return annotations

def volley_read_dataset(path, seqs):
  data = {}
  for sid in seqs:
    data[sid] = volley_read_annotations(path + '/%d/annotations.txt' % sid)
  return data

def volley_all_frames(data):
  frames = []
  for sid, anns in data.items():
    for fid, ann in anns.items():
      frames.append((sid, fid))
  return frames

def volley_random_frames(data, num_frames):
  frames = []
  for sid in np.random.choice(list(data.keys()), num_frames):
    fid = int(np.random.choice(list(data[sid]), []))
    frames.append((sid, fid))
  return frames

def volley_frames_around(frame, num_before=5, num_after=4):
  sid, src_fid = frame
  return [(sid, src_fid, fid)
          for fid in range(src_fid-num_before, src_fid+num_after+1)]
