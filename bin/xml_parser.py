import xml.etree.ElementTree as ET
import sys, os, glob
from fire import Fire



def main(annotations_dir):

    files = glob.glob(annotations_dir+'/*.xml')
    video = annotations_dir.split('/')[-1]
    print(video)
    with open('groundtruth.txt', 'w+') as output:
        for file in sorted(files):
            print(file.split('/')[-1])
            tree = ET.parse(file)
            root = tree.getroot()

            for child in root:
                for bbox in child.findall('bndbox'):
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    width = xmax-xmin
                    height = ymax-ymin
            output.write(str(xmin)+','+str(ymin)+','+str(width)+','+str(height)+'\n')


if __name__ == '__main__':
    Fire(main)


#/home/lv71186/deutsch/datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000000
#/home/lv71186/deutsch/datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00007009
#/home/lv71186/deutsch/datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00016000