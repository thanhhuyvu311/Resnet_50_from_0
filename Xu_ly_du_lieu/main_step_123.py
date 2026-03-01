import xml.etree.ElementTree as ET
import os,glob
import pandas as pd

if __name__ == '__main__':
    data_dir = '/home/huy/Documents/de_tai_tot_nghiep/Dataset-1-Thermal-Images.v1i.voc'
    anno_dir = os.path.join(data_dir,'anno') #lay dia chi thuc muc anno
    img_dir = os.path.join(data_dir,'imgs')
    #khoi tao dic gan nhan background voi id 0
    class_mapping = {'background':0}
    #tao 1 mang de chua cac gia tri sau khi doc ra vao day
    xml_list = []
    #lay dia chi file xml trong anno
    for xml_file in glob.glob(anno_dir + '/*.xml'):
        #print(xml_file)
        #lay vi tri cua file xml tren o dia
        tree = ET.parse(xml_file)
        #print(tree)
        #lay nut goc
        #dong dau tien trong file xml ban bat len se la nut goc.
        root = tree.getroot()
        #print(root)

        #lay kich thuoc anh
        #thong tin kich thuoc anh nam tai muc size
        width_img = int(root.find('size/width').text)
        height_img = int(root.find('size/height').text)
        channel_img = int(root.find('size/depth').text)
        path_img = os.path.join(img_dir,root.find('filename').text) #lay dia chi cung tung anh
        #print(path_img)
        # Xoa dau # tai print de xem no in ra gi
        #print(width_img,height_img,channel_img)

        #truy cap vao phan tu object de doc nhan, bbox
        for member in root.findall('object'):
            #tim ten class cua tung object
            class_name = member.find('name').text
            #print(class_name)
            class_id = class_mapping.setdefault(class_name,len(class_mapping))
            #print(class_id,class_name)

            bndbox = member.find('bndbox')
            if bndbox is not None:
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                w = xmax - xmin
                h = ymax - ymin
                x = xmin + w / 2
                y = ymin + h / 2
            else:
                w=h=x=y=0

            #tra ve (path_anh,x,y,w,h,class_id)
            value = (path_img,x,y,w,h,class_id)
            xml_list.append(value)
    column_name = ['path_img','x','y','w','h','class_id']
    xml_df = pd.DataFrame(xml_list,columns=column_name)

    #---gom cac bndbox trong cung 1 anh lai voi nhau vao 1 bien csv

    # tao 1 cot moi co ten bbox gom x y w h thanh 1 list [x,y,w,h] cho moi dong
    xml_df['bbox'] = xml_df.apply(lambda row: [row['x'],row['y'],row['w'],row['h']],axis=1)
    # gom nhom theo tung anh
    grouped_df = xml_df.groupby('path_img').agg({
        'bbox':list, #gom bbox thanh list
        'class_id':list #gom class_id thanh list
    }).reset_index() #reset_index de path_img tro thanh 1 cot binh thuong
    #luu file csv
    #luu y!: phai tao folder ten la csv_file trong thu muc dataset-1-.. truoc moi dung dc os.chdir
    os.chdir(data_dir+'/csv_file/')

    grouped_df.to_csv('data_information_grouped.csv',index=False)

