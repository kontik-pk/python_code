class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=DataLoader.collate_data, **kwargs)

    @staticmethod
    def collate_data(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, label_data, image_folder=None, transform=None):

        # CSV файл содержит: filename, width, height, class, xmin, ymin, xmax, ymax
        if os.path.isfile(label_data):
            self._csv = pd.read_csv(label_data)
        else:
            self._csv = xml_to_csv(label_data)

        #Если пути к картинкам нет, то картинки в той же самой директории, где и разметка
        if image_folder is None:
            self._root_dir = label_data
        else:
            self._root_dir = image_folder

        if transform is None:
            self.transform = default_transforms()
        else:
            self.transform = transform

    # Длина датасета
    def __len__(self):
        return len(self._csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Считываем имя изображения из первой колонки csv - файла (filename)
        #idx - индекс запроса
        img_name = os.path.join(self._root_dir, self._csv.iloc[idx, 0])
        image = read_image(img_name)

        # Координаты объекта -  xmin, ymin, xmax, and ymax
        box = self._csv.iloc[idx, 4:]
        box = torch.tensor(box).view(1, 4)

        # Метка
        label = self._csv.iloc[idx, 3]

        targets = {'boxes': box, 'labels': label}

        # Применяем трансформации
        if self.transform:
            width = self._csv.loc[idx, 'width']
            height = self._csv.loc[idx, 'height']

            # Применяем каждую трансформацию отдельно, потому что если в списке трансформаций есть
            # Resize или RandomHorizontalFlip, то их надо применить и к боксам
            updated_transforms = []
            scale_factor = 1.0
            random_flip = 0.0
            for t in self.transform.transforms:
                # каждую трансформацию добавляем в список
                updated_transforms.append(t)

                # Если есть уменьшение картинки, то масштабируем боксы
                if isinstance(t, transforms.Resize):
                    original_size = min(height, width)
                    scale_factor = original_size / t.size

                # Если есть horizontal flip, то применяем его к боксу тоже
                elif isinstance(t, transforms.RandomHorizontalFlip):
                    random_flip = t.p

            for t in updated_transforms:
                if isinstance(t, transforms.RandomHorizontalFlip):
                    if random.random() < random_flip:
                        image = transforms.RandomHorizontalFlip(1)(image)
                        # Flip box's x-coordinates
                        box[0, 0] = width - box[0, 0]
                        box[0, 2] = width - box[0, 2]
                        box[0, 0], box[0, 2] = box[0, (2, 0)]
                else:
                    image = t(image)

            # Масштабируем, если это необходимо
            targets['boxes'] = (box / scale_factor).long()

        return image, targets

def _to_device(images, targets):
  images = [image.to('cuda') for image in images]
  targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
  return images, targets

def _convert_to_int_labels(targets):
  for target in targets:
    target['labels'] = torch.tensor(int_mapping[target['labels']]).view(1)


