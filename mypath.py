class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'dataset_images':
            # folder that contains class labels
            root_dir = 'dataset_videos'

            # Save preprocess data into output_dir
            output_dir = 'VAR/dataset_images'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return 'c3d-pretrained.pth'
