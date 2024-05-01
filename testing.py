from flask import Flask,request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from singledataset import MyDataset
import io
from flask_cors import CORS
import statistics
import cv2
import numpy as np
app = Flask(__name__)
CORS(app, origins="http://localhost:3000", allow_headers=["Content-Type", "Authorization"])

def labels(val):
    if val == 0:
        ans = "No Smoke No Fire"
    elif val == 1:
        ans = "Fire with Smoke"
    else:
        ans = "Fire without Smoke"
    return ans

def process_image(imagergb,imageir,mode,modelss):
    models = str(modelss).strip("[]").replace("'","").split(',')
    # print(str(modelss))
    # print(models)
    flag=False
    result = "<table border=1><tr><th>Model Name</th><th>Result</th></tr>"
    bag=[]
    Model_list=["Logistic","MobilevNet"]
    print(len(models))
    for model in models:
        result+="<tr><td>"+model+"</td>"
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        # print("hi")
        if model=="ResNet" or model=="VGG Net":
            result+="<td>Coming Soon</td></tr>"
            continue
        elif model=="MobilevNet" and mode=="both":
            result+="<td>Coming Soon</td></tr>"
            continue

        path="saved_model//"+model+"//"+mode+".pkl"
        # path = "saved_model\Flame_one_stream_ir_0 batch=64.pkl"
        net = torch.load(path,map_location=torch.device("cpu"))
        net.eval()

        Dataset = MyDataset(imagergb,imageir,input_size=254,transform=False)
        dataloader = DataLoader(dataset=Dataset,batch_size=1)
        for (rgb,ir) in dataloader:
            if model=="Flame":
                print("hi")
                y_pre=net(rgb.to(DEVICE),ir.to(DEVICE),mode=mode)
            elif model in Model_list and mode!="both":
                y_pre = net(rgb)
            else:
                y_pre = net(rgb,ir,mode='both')
            _, label_index = torch.max(y_pre.data, dim=-1)
        val = label_index.item()
        bag.append(val)
        result+="<td>"+labels(val)+"</td></tr>"
    if len(bag)>1:
        result+="<tr><td>Ensemble</td><td><b>" + labels(statistics.mode(bag))+"</b></td></tr>"
        print(result)
    if (statistics.mode(bag)==1 or statistics.mode(bag)==2) and (mode=='ir' or mode=='both'):
        flag = True
    area_count=0
    total_area=0
    if flag:
        image = cv2.cvtColor(np.array(imageir), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 0, 0])    # Lower bound for red color
        upper_red = np.array([60, 255, 255])   # Upper bound for red color
        mask = cv2.inRange(hsv, lower_red, upper_red)
        mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        output_image = np.zeros_like(image)
        output_image[mask_binary == 255] = [255, 255, 255]
        print("Shape of mask_binary:", output_image.shape)
        single_channel_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
        print("Shape of mask_binary:", single_channel_gray.shape)
        num_labels, labelss, stats, centroids = cv2.connectedComponentsWithStats(single_channel_gray, connectivity=8)
        print("Number of areas:", num_labels - 1) 
        labeled_image = np.zeros_like(labelss, dtype=np.uint8)
        total_area=0
        area_count=0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # print("Area of component", i, ":", area)
            if area > 50:
                labeled_image[labelss == i] =255
                total_area+=area
                area_count+=1           
        print("Number of areas under fire",area_count)
        print("Total area under fire",total_area)
        cv2.imwrite("C:/Users/apuru/OneDrive/Desktop/ProjectSeminar/code/project/src/output1.jpg",labeled_image)

    return result+"</table>",flag,area_count,total_area
        
    
@app.route('/api/process_image', methods=['POST'])
def process_image_api():
    area_count=0    
    total_area=0
    if 'imagergb' not in request.files and 'imageir' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    # images = []
    print(len(request.files))
    # for key in request.files:
    #     image_file = request.files[key]
    #     print("Found")
    #     if image_file.filename == '':
    #         return jsonify({'error': 'No image provided'}), 400
    #     try:
    #         image = Image.open(io.BytesIO(image_file.read()))
    #         print(image_file.filename)
    #         images.append(image)
    #     except Exception as e:
    #         return jsonify({'error': str(e)}), 500
    data=request.form
    ans="in"
    print(data.get('models'))
    
    if data.get('mode') == 'rgb':
        try:
            image_file = request.files["imagergb"]
            image = Image.open(io.BytesIO(image_file.read()))
            imagergb = image
            ans,flag,area_count,total_area = process_image(imagergb,imagergb,data.get('mode'),data.get('models'))
        except Exception as e:
            return jsonify({'error':str(e)}), 500
    elif data.get('mode') == 'ir':
        try:
            image_file = request.files["imageir"]
            image = Image.open(io.BytesIO(image_file.read()))
            imageir = image
            ans,flag,area_count,total_area = process_image(imageir,imageir,"ir",data.get('models'))
        except Exception as e:
            return jsonify({'error':str(e)}), 500
    else:
        try:
            # print('hi')
            image_file = request.files["imageir"]
            image = Image.open(io.BytesIO(image_file.read()))
            imageir = image
            image_file = request.files["imagergb"]
            image = Image.open(io.BytesIO(image_file.read()))
            imagergb = image
            ans,flag,area_count,total_area = process_image(imagergb,imageir,"both",data.get('models'))
        except Exception as e:
            return jsonify({'error':str(e)}), 500

    # process_image(images,mode,model)
    # if image_file.filename == '':
    #     return jsonify({'error': 'No image provided'}), 400
    # try:
    #     print("hi")
    #     #image = Image.open(io.BytesIO(image_file.read()))
    #     #process_image(image,data.get('mode'),data.get('models'))
    #     return jsonify({'mode':str(data.get('mode')),'models':str(data.get('models')),'no':str(len(images))}), 200
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500
    return jsonify({"ma":"okay","ans":str(ans),"fire":flag,"total_area":str(total_area),"area_count":str(area_count)}),200

if __name__ == '__main__':
    app.run(host = 'localhost', port = 5000, debug = True)




class SingleImageDataset(torch.utils.data.Dataset):
        def __init__(self, image):
            self.image = image

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return self.image