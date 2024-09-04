import math
import os
import requests
import asyncio
import concurrent.futures
import multiprocessing as mp
import cv2
import numpy as np
import shapely.geometry
import re
import rasterio
import pandas as pd

def latlon2px(z,lat,lon):
    x = 2**z*(lon+180)/360*256
    y = -(.5*math.log((1+math.sin(math.radians(lat)))/(1-math.sin(math.radians(lat))))/math.pi-1)*256*2**(z-1)
    return x,y
def latlon2xy(z,lat,lon):
    x,y = latlon2px(z,lat,lon)
    x = int(x/256)
    y = int(y/256)
    return x,y


class imagedownload:
    def __init__(self, folder_path, city, zoom, file_path):
        self.folder_path = folder_path
        self.city = city
        self.zoom = zoom
        self.file_path = file_path
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        if not os.path.exists(self.folder_path + '/' +  self.city ):
            os.mkdir(self.folder_path + '/' +  self.city )
        if not os.path.exists(self.folder_path +  '/' +  self.city  + '/' + 'satellite_image'):                
            os.mkdir(self.folder_path +   '/' +  self.city +  '/satellite_image')

        
    def N2C(self,xtile, ytile, zoom):
        n = 2.0**zoom
        long_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = lat_rad * 180.0 / math.pi
        return lat_deg, long_deg

    def tifDownloader(self, url, x, y, z, headers, typeOfImage, image_name):
        xmax = self.N2C(x+1,y+1,z)[1]
        xmin = self.N2C(x,y,z)[1]
        ymax = self.N2C(x,y,z)[0]
        ymin = self.N2C(x+1,y+1,z)[0]       
        try: 
            req = requests.get(url, headers=headers, timeout=10)
            im_rgb = cv2.imdecode(np.frombuffer(req.content, np.uint8), -1)
            im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
            transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, 256, 256)
            # print(self.folder_path, typeOfImage)
            
            # print(self.folder_path + '/' + self.city + '/' + typeOfImage + '/' + image_name)
            if not os.path.exists(self.folder_path + image_name):
                src = rasterio.open(self.folder_path + '/' + self.city + '/' + typeOfImage + '/' + image_name, 'w', driver='GTiff',
                        height = 256, width = 256,
                        count=3, dtype=rasterio.uint8,
                        crs= "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0",
                        transform =  transform)
                profile = src.profile
                profile['photometric'] = "RGB"                            
                profile['tiled'] = True                            
                src.write(np.transpose(im_rgb, (2, 0, 1)))  
                src.close()              
        except:
            pass
        
    def imageDownloader(self, x, y, z, image_name, headers):
        # satellite image tile
        url_g_satellite = "http://mt1.google.com/vt?lyrs=s@162000000&hl=en&x=%d&y=%d&z=%d" % (x, y, self.zoom)
        # print(url_g_satellite)
        self.tifDownloader(url_g_satellite,x,y,z, headers, "satellite_image", image_name)


    def xy_generator(self, files): 
        user_agent = 'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; de-at) AppleWebKit/533.21.1 (KHTML, like Gecko) Version/5.0.5 Safari/533.21.1'
        headers = {'User-Agent': user_agent}
        delimiters = "/", "_", "."
        regexpattern = '|'.join(map(re.escape, delimiters))
        for file in files:
            split_path = re.split(regexpattern, file)
            y = int(split_path[-2])
            x = int(split_path[-3])
            z = int(split_path[-4])
            yield x, y, z, file, headers

    async def async_download(self, flgen):
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(
                    executor, 
                    self.imageDownloader,
                    x, y, z, image_name, headers
                ) for x, y, z, image_name, headers in flgen
            ]
            for response in await asyncio.gather(*futures):
                pass

    def download(self):
        file_list = pd.read_csv(self.file_path)
        flgen = self.xy_generator(file_list['Key'])
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.async_download(flgen))    
        loop.close()

# async def async_upload(flgen):
#         with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
#             loop = asyncio.get_event_loop()
#             futures = [
#                 loop.run_in_executor(
#                     executor, 
#                     self.imageDownloader,
#                     x, y, z, image_name, headers
#                 ) for x, y, z, image_name, headers in flgen
#             ]
#             for response in await asyncio.gather(*futures):
#                 pass

def filenames_to_csv(zoom, lat_start, lon_start, lat_stop, lon_stop, path, prefix=None):
    start_x, start_y = latlon2xy(zoom, lat_start, lon_start)
    stop_x, stop_y = latlon2xy(zoom, lat_stop, lon_stop)
    file_list = []
    for x in range(start_x, stop_x + 1):
        for y in range(start_y, stop_y + 1):
    # for x in range(start_x, start_x+2):
    #     for y in range(start_y, start_y+5):
            if prefix != None: 
                image_name = prefix + "%d_%d_%d.tif" % (zoom, x, y)
            else:
                image_name = "%d_%d_%d.tif" % (zoom, x, y)
            file_list.append(image_name)
    
    filenames = pd.DataFrame(file_list, columns=['Key'])
    filenames.to_csv(path)


# 40.795200, -73.961619
# 40.788973, -73.955274
if __name__ == "__main__":
    zoom = 19
    # -81.565204,26.769081,-80.87247,27.210638
    xstart = list()
    ystart = list()
    xstop = list()
    ystop = list()
    total = list()
    boundaries = dict()
    lat_start, lon_start = 32.322702, -89.163724
    lat_stop, lon_stop = 32.319914, -89.160942





    # lat_start = 40.795200
    # lat_stop = 40.788973
    # lon_start = -73.961619
    # lon_stop = -73.955274
    # lat_start = 33.768044
    # lat_stop = 33.726199
    # lon_start = -118.281341
    # lon_stop = -118.182579

    start_x, start_y = latlon2xy(zoom, lat_start, lon_start)
    stop_x, stop_y = latlon2xy(zoom, lat_stop, lon_stop)
    total_images = int((stop_x - start_x) * (stop_y - start_y))
    folder_path = './'
    city = 'newton'
    # xstart.append(start_x)
    # xstop.append(stop_x)
    # ystart.append(start_y)
    # ystop.append(stop_y)
    # total.append(total_images)
    print(total_images)
    # print(stop_x, stop_y)
    filenames_to_csv(zoom, lat_start, lon_start, lat_stop, lon_stop, "./glades.csv")
    downloadimages = imagedownload(folder_path=folder_path, city=city, zoom=19, file_path=folder_path + 'glades.csv')
    downloadimages.download()
    
