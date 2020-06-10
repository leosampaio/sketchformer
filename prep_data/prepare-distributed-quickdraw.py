import glob
import argparse
import os
import numpy as np


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir')
    parser.add_argument('--class-list', type=str, default='prep_data/quickdraw/list_quickdraw.txt')
    parser.add_argument('--n-chunks', type=int, default=10)
    parser.add_argument('--n-classes', type=int, default=10)
    parser.add_argument('--cut-chunks', type=int, default=0)
    parser.add_argument('--target-dir')

    args = parser.parse_args()

    class_names = []
    with open(args.class_list) as clf:
        class_names = clf.read().splitlines()

    class_files = []
    for class_name in class_names:
        file = "{}/{}.npz".format(args.dataset_dir, class_name)
        class_files.append(file)

    class_names = class_names[:args.n_classes]
    class_files = class_files[:args.n_classes]

    target_basename = os.path.join(args.target_dir, "train_{:03}.npz")
    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    n_chunks = args.n_chunks
    did_read_test_and_validation = False
    val_set, test_set = None, None
    val_y, test_y = None, None
    means, stds = 0, 0
    n_samples_train, n_samples_test, n_samples_valid = 0, 0, 0

    # preload the classes into an array
    all_data = []
    for i, (class_name, data_filepath) in enumerate(zip(class_names, class_files)):
        print("Loading data from class {} ({}/{})...".format(
            class_name, i + 1, len(class_files)))
        all_data.append(np.load(data_filepath, encoding='latin1', allow_pickle=True))

    for chunk in range(0, n_chunks - args.cut_chunks):

        cur_train_set, cur_train_y = None, None

        # collect a bit of each class
        for i, (class_name, data_filepath) in enumerate(zip(class_names, class_files)):
            data = all_data[i]
            print("Loading chunk {}/{} from class {} ({}/{})...".format(
                chunk + 1, n_chunks, class_name, i + 1, len(class_files)))

            n_samples = len(data['train']) // n_chunks
            start = n_samples * chunk
            end = start + n_samples
            samples = data['train'][start:end]
            labels = np.ones((n_samples,), dtype=int) * i

            cur_train_set = np.concatenate((
                cur_train_set, samples)) if cur_train_set is not None else samples
            cur_train_y = np.concatenate((
                cur_train_y, labels)) if cur_train_y is not None else labels

            if not did_read_test_and_validation:
                total_val = int(((n_chunks - args.cut_chunks) / n_chunks) * len(data['valid']))
                total_test = int(((n_chunks - args.cut_chunks) / n_chunks) * len(data['test']))
                val_set = np.concatenate((val_set, data['valid'][:total_val])) if val_set is not None else data['valid'][:total_val]
                test_set = np.concatenate((test_set, data['test'][:total_test])) if test_set is not None else data['test'][:total_test]
                val_y = np.concatenate((
                    val_y, np.ones((total_val,), dtype=int) * i)) if val_y is not None else np.ones((total_val,), dtype=int) * i
                test_y = np.concatenate((
                    test_y, np.ones((total_test,), dtype=int) * i)) if test_y is not None else np.ones((total_test,), dtype=int) * i

        # compute the local mean and standard dev
        data = []
        for sketch in cur_train_set:
            for delta in sketch:
                data.append(delta[:2])
        std, mean = np.std(data), np.mean(data)
        means += mean
        stds += std

        # create .npz file with the complete chunk
        print("Saving chunk {}/{}...".format(chunk + 1, n_chunks))
        np.savez(target_basename.format(chunk),
                 x=cur_train_set,
                 y=cur_train_y,
                 label_names=class_names,
                 std=std,
                 mean=mean)

        # save the .npz for test and validation sets
        if not did_read_test_and_validation:
            did_read_test_and_validation = True
            valid_f = os.path.join(args.target_dir, "valid.npz")
            test_f = os.path.join(args.target_dir, "test.npz")
            np.savez(valid_f,
                     x=val_set,
                     y=val_y,
                     label_names=class_names)
            np.savez(test_f,
                     x=test_set,
                     y=test_y,
                     label_names=class_names)
            n_samples_test = len(test_set)
            n_samples_valid = len(val_set)

        n_samples_train += len(cur_train_set)

    # finally, save the gathered metadata
    meta_f = os.path.join(args.target_dir, "meta.npz")
    np.savez(meta_f,
             std=stds / n_chunks,
             mean=means / n_chunks,
             class_names=class_names,
             n_classes=len(class_names),
             n_samples_train=n_samples_train,
             n_samples_test=n_samples_test,
             n_samples_valid=n_samples_valid)


if __name__ == '__main__':
    main()
