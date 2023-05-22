import os, random, pickle, json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt



class ImageUtils():
    '''Various image utilities.'''
    def __init__(self, config):
        self.config = config
    
    
    def show_image(self, im_tensor):
        '''Display an image'''
        #unswap the color channel from index 0 (needed for model) back to index 2
        im_tensor = torch.swapaxes(im_tensor, 0, 2)
        #denormalize
        im_tensor = ((im_tensor * self.config.imnet_std) + self.config.imnet_mean) * 255
        plt.imshow(im_tensor.numpy().astype(np.int32))
        plt.show()
        
    
    def flip_augment_class(self, im_tensors, y, im_names, class_idx_to_flip=1, num=0):
        '''Augment a class of images by flipping them horizontally'''
        #get target class instances
        trg_class_idx = (y==class_idx_to_flip).nonzero(as_tuple=True)[0]
        if num:
            trg_class_idx = trg_class_idx[:num]
        trg_class_tensors = im_tensors[trg_class_idx]
        trg_im_names = [im_names[i] for i in trg_class_idx.tolist()]
        im_names.extend(trg_im_names)
        
        print('Flipping %d instances of class index %d' % (trg_class_tensors.size(0), class_idx_to_flip))
        
        #shape (3 x H x W), flip W, index 2
        trg_class_flipped_tensors = torch.flip(trg_class_tensors, dims=(2,))
        
        #add flipped images and targets
        im_tensors = torch.cat((im_tensors, trg_class_flipped_tensors), 0)
        y = torch.cat((y, torch.ones(trg_class_flipped_tensors.size(0)) * class_idx_to_flip))
        print('New dataset size: %s' % im_tensors.size(0))
        
        #shuffle all
        print('Shuffling...')
        idx = torch.randperm(im_tensors.size(0))
        im_tensors = im_tensors[idx]
        y = y[idx]

        return im_tensors, y, im_names
    
    
    def balance_classes(self, im_tensors, y):
        '''Trim class instances to that of the minimum class to get a balanced set'''
        class_labs, cts = np.unique(y, return_counts=True)
        minority_ct = cts.min()
        print('Truncating all class counts to min class count %d' % minority_ct)
        
        class_x = []
        class_y = []
        for class_lab in class_labs.tolist():
            trg_class_idx = (y==class_lab).nonzero(as_tuple=True)[0][:minority_ct]
            class_x.append(im_tensors[trg_class_idx])
            class_y.append(torch.ones(trg_class_idx.size(0)) * class_lab)
            
        balanced_x = torch.cat(class_x, 0)
        balanced_y = torch.cat(class_y, 0)
        
        print('Shuffling...')
        idx = torch.randperm(balanced_x.size(0))
        balanced_x = balanced_x[idx]
        balanced_y = balanced_y[idx]
        
        print('X: %s, y: %s' % (balanced_x.size(0), balanced_y.size(0)))
        
        return balanced_x, balanced_y
    
    
    def resize_image_tensors(self, xs, new_size):
        '''Resize the image tensors for a model with specific required dimensions'''
        xs_resized = []
        
        for i in range(xs.shape[0]):
            x_im = xs[i]
            x_im = torch.swapaxes(x_im, 0, 2)
            x_resized = cv2.resize(x_im.numpy(), (new_size, new_size), interpolation = cv2.INTER_LINEAR)
            x_resized = torch.swapaxes(torch.from_numpy(x_resized), 0, 2)
            xs_resized.append(x_resized)
            
        return torch.stack(xs_resized)
    
        
        
class WWMRImageProcessor():
    '''WWMR DB face mask image processing'''
    def __init__(self, config):
        self.config = config
        # we're not going to use MRHN (hanging from ear) because the mask isn't even visible in some crops
        self.mask_stats = set(['MRCW', 'MRFH', 'MRNC', 'MRNN', 'MRNW', 'MRTN', 'MSFC'])
        
        #map these to lab2idx = {'without_mask':0, 'with_mask':1, 'mask_weared_incorrect':2}
        self.lab2idx_3class = {'MRCW': 1, 'MRFH': 2, 'MRNC': 2, 'MRNN': 2, 'MRNW': 0, 'MRTN': 2, 'MSFC': 2}
        
        
    def process_images(self):
        '''Prepare the WWMR images and store image tensors, targets, and image names.'''
        if os.path.exists('%s/x_%d.pt' % (self.config.wwmr_path, self.config.im_size)):
            print('Images already processed, delete %s/x_%d.pt to reprocess' % (self.config.wwmr_path, self.config.im_size))
            return 
        
        if os.path.exists('%s/im_data.json' % self.config.wwmr_path):
            with open('%s/im_data.json' % self.config.wwmr_path, 'r') as f:
                im_data = json.load(f)
        else:
            print('Loading image data...')
            im_data = self.__get_image_data(self.config.wwmr_ims_path1)
            im_data.update(self.__get_image_data(self.config.wwmr_ims_path2))

            print('Getting bounding box info...')
            self.__get_bounding_boxes(im_data)

            print('Writing image data to %s/im_data.json' % self.config.wwmr_path)
            with open('%s/im_data.json' % self.config.wwmr_path, 'w') as o:
                json.dump(im_data, o)
            
        print('Cropping and resizing and storing tensors...')
        self.__crop_and_resize(im_data)
        
        print('Done!')
        
        
    def __get_image_data(self, ims_path):
        '''Get a dictionary of image sizes mapped to list of image paths'''
        im_data = {}

        i=0
        # image name encodes the following (if present): subjnum_maskstat_masktype_headangle_maskinbackgroundforeground.jpg
        for path, _, fns in os.walk(ims_path):
            for fn in fns:
                if fn.lower().endswith('.jpg'):
                    im_name = fn[:-4]
                elif fn.lower().endswith('.jpeg'):
                    im_name = fn[:-5]
                else:
                    continue

                items = im_name.split('_')

                if len(items)<3 or len(items)>5:
                    print(items)
                    return {}

                items = items[1:]
                mask_stat = ''

                for item in items:
                    if item in self.mask_stats:
                        mask_stat = item

                if not mask_stat:
                    continue

                fp = '%s/%s' % (path, fn)

                #read the image with opencv2
                try:
                    im = cv2.imread(fp)
                except Exception as ex:
                    print('couldnt open %s: %s' % (fp, str(ex)))
                    continue

                if im is None:
                    print('couldnt open %s' % (fp))
                    continue

                #get and store the image size
                sz = im.shape

                im_data[im_name] = {'path':fp, 
                                      'size':sz,
                                      'mask_status':mask_stat
                                   }

                i+=1
                if i%100==0:
                    print('\n', i, im_data[im_name])

        print('%s files processed' % i)

        return im_data
    
    
    def __get_bounding_boxes(self, im_data):
        '''Get the face bounding box info for the images from the provided annotation file'''
        for path, _, fns in os.walk(self.config.wwmr_labs_path):
            for fn in fns:
                if not fn.endswith('.txt'):
                    continue
                if fn.startswith('classes'):
                    continue

                im_name = fn[:-4]

                if not im_name in im_data:
                    print('%s not in im sizes' % im_name)
                    continue

                fp = '%s/%s' % (path, fn)

                x, y, w, h = 0,0,0,0
                with open(fp, 'r') as f:
                    for line in f:
                        if not line.startswith('0'):
                            continue

                        _, x, y, w, h = line.replace('\n','').split(' ')
                        break

                if not w:
                    print('full bounding box not found for %s' % fn)
                    continue

                im_data[im_name]['bb'] = {'x':x, 'y':y, 'w':w, 'h':h}
                    
                    
    def __crop_and_resize(self, im_data):
        '''Crop and resize the images'''
        im_arrays = []
        im_names = []
        y_3class = []

        i=0
        for im_name, im_dict in im_data.items():
            i+=1
            if i%100==0:
                print(i)
                
            im = cv2.cvtColor(cv2.imread(im_dict['path']), cv2.COLOR_BGR2RGB)
            sz = im.shape

            h = int(sz[0]*float(im_dict['bb']['h']))
            w = int(sz[1]*float(im_dict['bb']['w']))

            y1 = int(sz[0]*float(im_dict['bb']['y'])) - h//2
            x1 = int(sz[1]*float(im_dict['bb']['x'])) - w//2
            x2 = x1 + w
            y2 = y1 + h

            crop_w = x2-x1
            crop_h = y2-y1

            if crop_w < crop_h:
                pad = (crop_h - crop_w)//2
                x1 = max(0, x1-pad)
                x2 = min(x2+pad, sz[1]-1)
            elif crop_w > crop_h:
                pad = (crop_w - crop_h)//2
                y1 = max(0, y1-pad)
                y2 = min(y2+pad, sz[0]-1)

            im_crop = im[y1:y2, x1:x2, :]

            #resize the image
            im_new = cv2.resize(im_crop, (self.config.im_size, self.config.im_size), interpolation = cv2.INTER_LINEAR)

            im_arrays.append(im_new)
            im_names.append(im_name)
            y_3class.append(self.lab2idx_3class[im_dict['mask_status']])

        # The images have to be loaded in to a range of [0, 1] and then normalized using
        # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        im_arrays = np.reshape(im_arrays, (-1, self.config.im_size, self.config.im_size, 3))

        im_arrays = torch.from_numpy(im_arrays)
        
        #normalize
        im_arrays = ((im_arrays/255) - self.config.imnet_mean) / self.config.imnet_std
        
        #need to swap the RGB dim from last to second index
        im_arrays = torch.swapaxes(im_arrays, 3, 1)
        print(im_arrays.shape)

        print('saving %s/x_%d.pt' % (self.config.wwmr_path, self.config.im_size))
        torch.save(im_arrays, '%s/x_%d.pt' % (self.config.wwmr_path, self.config.im_size))
        
        if not os.path.exists('%s/y.pt' % (self.config.wwmr_path)):
            torch.save(torch.LongTensor(y_3class), '%s/y.pt' % (self.config.wwmr_path))

            with open('%s/im_names.txt' % self.config.wwmr_path, 'w') as o:
                o.write('\n'.join(im_names))
                
            
    def load_data(self):
        '''Load the processed image tensors, targets, and file names'''
        x = torch.load('%s/x_%d.pt' % (self.config.wwmr_path, self.config.im_size))

        with open('%s/im_names.txt' % self.config.wwmr_path, 'r') as f:
            im_names = f.read().split('\n')

        y = torch.load('%s/y.pt' % (self.config.wwmr_path)).type(torch.LongTensor)

        return x, y, im_names
            
            
    def show_image(self, im_index, im_arrays, im_names, y_3class):
        '''Display an image'''
        self.show_image(im_arrays[im_index], im_names[im_index], y_3class[im_index])
        
        
    def show_image(self, im_tensor, im_name, y_3class):
        '''Display an image'''
        mask_status = self.config.idx2lab[y_3class.item()]

        #unswap the color channel from index 0 (needed for model) back to index 2
        im_tensor = torch.swapaxes(im_tensor, 0, 2)
                        
        #denormalize
        im_tensor = ((im_tensor * self.config.imnet_std) + self.config.imnet_mean) * 255

        print('Image name: %s' % im_name)
        print('Mask status: %s' % mask_status)

        plt.imshow(im_tensor.numpy().astype(np.int32))
        plt.show()
        
        
        
class FMDImageProcessor():
    '''Face Mask Detection dataset processing'''
    def __init__(self, config):
        self.config = config
        self.crop_pad = 20
        
        
    def process_images(self, min_crop_dim=70):
        '''Prepare the FMD images and store image tensors, targets, and image names.'''
        self.__extract_and_resize_faces(min_crop_dim)
        
        
    def __extract_and_resize_faces(self, min_crop_dim):
        '''Use the provided bounding box information to extract and resize the faces.'''
        im_arrays = []
        im_names = []
        y = []

        i=0
        for im_name in os.listdir(self.config.fmd_ims_path):
            i+=1
            if i%50==0:
                print('Image %d, cropped faces: %d' % (i, len(y)))

            im_path = '%s/%s' % (self.config.fmd_ims_path, im_name)
            annot_path = '%s/%s.xml' % (self.config.fmd_annots_path, im_name[:-4])

            bbs = self.__get_bbs(annot_path)

            im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
            sz = im.shape

            for lab_idx, x1, y1, x2, y2 in bbs:
                crop_w = x2-x1
                crop_h = y2-y1
                max_crop_dim = max(crop_w, crop_h)

                if max_crop_dim < min_crop_dim:
                    continue

                if x1-self.crop_pad >=0 and x2+self.crop_pad<sz[1] and y1-self.crop_pad*2.5>=0 and y2+self.crop_pad*0.5<sz[0]:
                    x1-=self.crop_pad
                    x2+=self.crop_pad
                    y1-=int(self.crop_pad*2.5)
                    y2+=int(self.crop_pad*0.5)
                    crop_w = x2-x1
                    crop_h = y2-y1
                    max_crop_dim = max(crop_w, crop_h)

                if crop_w < crop_h:
                    pad = (crop_h - crop_w)//2
                    x1 = max(0, x1-pad)
                    x2 = min(x2+pad, sz[1]-1)
                elif crop_w > crop_h:
                    pad = (crop_w - crop_h)//2
                    y1 = max(0, y1-pad)
                    y2 = min(y2+pad, sz[0]-1)

                im_crop = im[y1:y2, x1:x2, :]

                #resize the image
                im_new = cv2.resize(im_crop, (self.config.im_size, self.config.im_size), interpolation = cv2.INTER_LINEAR)

                im_arrays.append(im_new)
                im_names.append(im_name)
                y.append(lab_idx)

        #reshape the data as needed by the models
        im_arrays = np.reshape(im_arrays, (-1, self.config.im_size, self.config.im_size, 3))

        im_arrays = torch.from_numpy(im_arrays)
                        
        #normalize
        im_arrays = ((im_arrays/255) - self.config.imnet_mean) / self.config.imnet_std
                        
        im_arrays = torch.swapaxes(im_arrays, 3, 1)
        print(im_arrays.shape)

        print('saving %s/x_%d.pt' % (self.config.fmd_path, self.config.im_size))
        torch.save(im_arrays, '%s/x_%d.pt' % (self.config.fmd_path, self.config.im_size))
        torch.save(torch.LongTensor(y), '%s/y.pt' % (self.config.fmd_path))
        with open('%s/im_names.txt' % self.config.fmd_path, 'w') as o:
            o.write('\n'.join(im_names))


    def __get_bbs(self, annot_path):
        '''Get the provided bounding box info'''
        bbs = []
        with open(annot_path) as f:
            bb = []
            for line in f:
                line = line.replace('\n','').strip()

                if line.startswith('</object>'):
                    bbs.append(bb.copy())
                    bb = []
                elif line.startswith('<name>'):
                    lab = line.replace('<name>','').replace('</name>','')
                    bb.append(self.config.lab2idx[lab])
                elif line.startswith('<xmin>'):
                    x1 = line.replace('<xmin>','').replace('</xmin>','')
                    bb.append(int(x1))
                elif line.startswith('<ymin>'):
                    y1 = line.replace('<ymin>','').replace('</ymin>','')
                    bb.append(int(y1))
                elif line.startswith('<xmax>'):
                    x2 = line.replace('<xmax>','').replace('</xmax>','')
                    bb.append(int(x2))
                elif line.startswith('<ymax>'):
                    y2 = line.replace('<ymax>','').replace('</ymax>','')
                    bb.append(int(y2))

        return bbs
    
    
    def load_data(self):
        '''Load the prepared data'''
        fmd_x = torch.load('%s/x_%d.pt' % (self.config.fmd_path, self.config.im_size))
        fmd_y = torch.load('%s/y.pt' % (self.config.fmd_path)).type(torch.LongTensor)
        with open('%s/im_names.txt' % self.config.fmd_path, 'r') as f:
            fmd_im_names = f.read().split('\n')

        return fmd_x, fmd_y, fmd_im_names
    
    
    def show_image(self, im_tensor):
        '''Display an image'''
        #unswap the color channel from index 0 (needed for model) back to index 2
        im_tensor = torch.swapaxes(im_tensor, 0, 2)
        
        #denormalize
        im_tensor = ((im_tensor * self.config.imnet_std) + self.config.imnet_mean) * 255
        
        plt.imshow(im_tensor.numpy().astype(np.int32))
        plt.show()
        
        
        
class MFNImageProcessor():
    '''Face Mask Detection dataset processing'''
    def __init__(self, config):
        self.config = config
        
        
    def process_images(self, num_ims=0):
        '''Prepare the MFN images and store image tensors, targets, and image names.'''
        mfn_x = []
        mfn_y = []
        mfn_names = []
        for fn in os.listdir(self.config.mfn_ims_path):
            im = cv2.cvtColor(cv2.imread('%s/%s' % (self.config.mfn_ims_path, fn)), cv2.COLOR_BGR2RGB)
            im_new = cv2.resize(im, (self.config.im_size, self.config.im_size), interpolation = cv2.INTER_LINEAR)
            
            mfn_x.append(im_new)
            mfn_y.append(self.config.lab2idx['with_mask'])
            mfn_names.append(fn)
            
            if len(mfn_x)%100==0:
                print(len(mfn_x))
                
            if num_ims and len(mfn_x)>=num_ims:
                break
                
        mfn_y = torch.tensor(mfn_y, dtype=torch.long)

        #reshape the data as needed by the models
        mfn_x = np.reshape(mfn_x, (-1, self.config.im_size, self.config.im_size, 3))
        mfn_x = torch.from_numpy(mfn_x)
        
        #normalize
        mfn_x = ((mfn_x/255) - self.config.imnet_mean) / self.config.imnet_std
        mfn_x = torch.swapaxes(mfn_x, 3, 1)
        
        print(mfn_x.shape)
        print('saving %s/x_%d.pt' % (self.config.mfn_path, self.config.im_size))
        torch.save(mfn_x, '%s/x_%d.pt' % (self.config.mfn_path, self.config.im_size))
        torch.save(mfn_y, '%s/y.pt' % (self.config.mfn_path))
        with open('%s/im_names.txt' % (self.config.mfn_path), 'w') as o:
            o.write('\n'.join(mfn_names))
            
            
    def load_data(self):
        '''Load the prepared data'''
        mfn_x = torch.load('%s/x_%d.pt' % (self.config.mfn_path, self.config.im_size))
        mfn_y = torch.load('%s/y.pt' % (self.config.mfn_path)).type(torch.LongTensor)
        with open('%s/im_names.txt' % self.config.mfn_path, 'r') as f:
            mfn_im_names = f.read().split('\n')

        return mfn_x, mfn_y, mfn_im_names
    
            
    def show_image(self, im_tensor):
        '''Show an image'''
        #unswap the color channel from index 0 (needed for model) back to index 2
        im_tensor = torch.swapaxes(im_tensor, 0, 2)
        
        #denormalize
        im_tensor = ((im_tensor * self.config.imnet_std) + self.config.imnet_mean) * 255
        
        plt.imshow(im_tensor.numpy().astype(np.int32))
        plt.show()
            

            
class GANImageProcessor():
    '''Face Mask Detection dataset processing'''
    def __init__(self, config):
        self.config = config
        
        
    def process_images(self, num_ims=0):
        '''Prepare the GAN images and store image tensors, targets, and image names.'''
        gan_x = []
        gan_y = []
        gan_names = []
        for fn in os.listdir(self.config.gan_ims_path):
            im = cv2.cvtColor(cv2.imread('%s/%s' % (self.config.gan_ims_path, fn)), cv2.COLOR_BGR2RGB)
            
            #resize the image
            im_new = cv2.resize(im, (self.config.im_size, self.config.im_size), interpolation = cv2.INTER_LINEAR)
            
            gan_x.append(im_new)
            gan_y.append(self.config.lab2idx['without_mask'])
            gan_names.append(fn)
            
            if len(gan_x)%100==0:
                print(len(gan_x))
                
            if num_ims and len(gan_x)>=num_ims:
                break
                
        gan_y = torch.tensor(gan_y, dtype=torch.long)

        #reshape the data as needed by the models
        gan_x = np.reshape(gan_x, (-1, self.config.im_size, self.config.im_size, 3))
        gan_x = torch.from_numpy(gan_x)
        
        #normalize the images
        gan_x = ((gan_x/255) - self.config.imnet_mean) / self.config.imnet_std
        gan_x = torch.swapaxes(gan_x, 3, 1)

        print('saving %s/x_%d.pt' % (self.config.gan_path, self.config.im_size))
        torch.save(gan_x, '%s/x_%d.pt' % (self.config.gan_path, self.config.im_size))
        torch.save(gan_y, '%s/y.pt' % (self.config.gan_path))
        with open('%s/im_names.txt' % (self.config.gan_path), 'w') as o:
            o.write('\n'.join(gan_names))
            
            
    def load_data(self):
        '''Load the prepared data'''
        gan_x = torch.load('%s/x_%d.pt' % (self.config.gan_path, self.config.im_size))
        gan_y = torch.load('%s/y.pt' % (self.config.gan_path)).type(torch.LongTensor)
        with open('%s/im_names.txt' % self.config.gan_path, 'r') as f:
            gan_im_names = f.read().split('\n')

        return gan_x, gan_y, gan_im_names
    
            
    def show_image(self, im_tensor):
        '''Display an image'''
        #unswap the color channel from index 0 (needed for model) back to index 2
        im_tensor = torch.swapaxes(im_tensor, 0, 2)
        
        #denormalize
        im_tensor = ((im_tensor * self.config.imnet_std) + self.config.imnet_mean) * 255
        
        plt.imshow(im_tensor.numpy().astype(np.int32))
        plt.show()
                    
            
                    