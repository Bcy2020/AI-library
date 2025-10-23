#include <omp.h>
#include "../DataBase.h"
#include"../NeuralNetwork.h"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>

// ���Ͷ��壺���ڶ�ȡ�ļ�ͷ��32λ�޷�������
using uchar = unsigned char;
using uint32 = uint32_t;

// ��ȡMNIST�ļ��е�32λ������MNISTʹ�ô���ֽ���
uint32 read_uint32(std::ifstream& file) {
    uint32 result;
    uchar bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    // ת������ֽ��򵽱����ֽ���
    result = (static_cast<uint32>(bytes[0]) << 24) |
        (static_cast<uint32>(bytes[1]) << 16) |
        (static_cast<uint32>(bytes[2]) << 8) |
        static_cast<uint32>(bytes[3]);
    return result;
}

// ����ͼ���ļ������ݿ�
bool load_images(const std::string& filename, DataBase& db, int label_count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "�޷���ͼ���ļ�: " << filename << std::endl;
        return false;
    }

    // ��ȡͼ���ļ�ͷ
    uint32 magic = read_uint32(file);
    uint32 count = read_uint32(file);
    uint32 rows = read_uint32(file);
    uint32 cols = read_uint32(file);

    // ��֤MNISTͼ���ļ�ħ��
    if (magic != 2051) {
        std::cerr << "��Ч��ͼ���ļ�ħ��: " << magic << std::endl;
        return false;
    }

    // ���ͼ���������ǩ�����Ƿ�ƥ��
    if (count != static_cast<uint32>(label_count)) {
        std::cerr << "ͼ���������ǩ������ƥ��" << std::endl;
        return false;
    }

    // ��ȡͼ�����ݲ����ӵ����ݿ�
    const int pixel_count = rows * cols;
    std::vector<uchar> buffer(pixel_count);

    for (uint32 i = 0; i < count; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), pixel_count);

        Data_pair data;
        data.input = VectorXd(pixel_count);
        // ����ֵ��һ����[0, 1]
        for (int j = 0; j < pixel_count; ++j) {
            data.input[j] = static_cast<double>(buffer[j]) / 255.0;
        }

        db.push_back(data);
    }

    std::cout << "�ɹ����� " << count << " ��ͼ�� (" << rows << "x" << cols << ")" << std::endl;
    return true;
}

// ���ر�ǩ�ļ���ƥ�䵽���ݿ�
bool load_labels(const std::string& filename, DataBase& db) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "�޷��򿪱�ǩ�ļ�: " << filename << std::endl;
        return false;
    }

    // ��ȡ��ǩ�ļ�ͷ
    uint32 magic = read_uint32(file);
    uint32 count = read_uint32(file);

    // ��֤MNIST��ǩ�ļ�ħ��
    if (magic != 2049) {
        std::cerr << "��Ч�ı�ǩ�ļ�ħ��: " << magic << std::endl;
        return false;
    }

    // ����ǩ���������ݿ��С�Ƿ�ƥ��
    if (count != static_cast<uint32>(db.size())) {
        std::cerr << "��ǩ������ͼ��������ƥ��" << std::endl;
        return false;
    }

    // ��ȡ��ǩ���ݲ����õ����ݿ�
    for (uint32 i = 0; i < count; ++i) {
        uchar label;
        file.read(reinterpret_cast<char*>(&label), 1);

        // ʹ�ö��ȱ����ʾ��ǩ��0-9��10�����
        db.Datas[i].target = VectorXd::Zero(10);
        db.Datas[i].target[label] = 1.0;
    }

    std::cout << "�ɹ����� " << count << " ����ǩ" << std::endl;
    return true;
}

// ����ѵ��������
bool load_training_set(DataBase& train_db, const std::string& base_path) {
    std::string img_path = base_path + "train-images.idx3-ubyte";
    std::string lbl_path = base_path + "train-labels.idx1-ubyte";

    // �ȶ�ȡ��ǩ��ȡ������������֤ͼ��������
    std::ifstream lbl_file(lbl_path, std::ios::binary);
    if (!lbl_file.is_open()) {
        std::cerr << "�޷���ѵ����ǩ�ļ�: " << lbl_path << std::endl;
        return false;
    }
    read_uint32(lbl_file); // ����ħ��
    uint32 train_count = read_uint32(lbl_file);
    lbl_file.close();

    // ����ͼ��ͱ�ǩ
    if (!load_images(img_path, train_db, train_count)) return false;
    if (!load_labels(lbl_path, train_db)) return false;

    return true;
}

// ���ز��Լ�����
bool load_test_set(DataBase& test_db, const std::string& base_path) {
    std::string img_path = base_path + "t10k-images.idx3-ubyte";
    std::string lbl_path = base_path + "t10k-labels.idx1-ubyte";

    // �ȶ�ȡ��ǩ��ȡ����
    std::ifstream lbl_file(lbl_path, std::ios::binary);
    if (!lbl_file.is_open()) {
        std::cerr << "�޷��򿪲��Ա�ǩ�ļ�: " << lbl_path << std::endl;
        return false;
    }
    read_uint32(lbl_file); // ����ħ��
    uint32 test_count = read_uint32(lbl_file);
    lbl_file.close();

    // ����ͼ��ͱ�ǩ
    if (!load_images(img_path, test_db, test_count)) return false;
    if (!load_labels(lbl_path, test_db)) return false;

    return true;
}

// ��ʾ���ݼ�������Ϣ
void print_dataset_info(DataBase& train_db, DataBase& test_db) {
    std::cout << "\n���ݼ���Ϣ:" << std::endl;
    std::cout << "ѵ������С: " << train_db.size() << std::endl;
    std::cout << "���Լ���С: " << test_db.size() << std::endl;
    if (train_db.size() > 0) {
        std::cout << "����ά��: " << train_db.Datas[0].input.size() << std::endl;
        std::cout << "���ά��: " << train_db.Datas[0].target.size() << std::endl;
    }
}

int main() {
    omp_set_num_threads(16);
    // ���ݼ�·��
    const std::string data_path = "C:\\Users\\Lenovo\\Desktop\\AI project\\Number classification\\";

    // ��ʼ�����ݿ⣨ѵ��������80%������ѵ�������ݿ���Ч��
    DataBase train_db(0.8);
    DataBase test_db(1.0);  // ���Լ������֣�ȫ�����ڲ���

    try {
        // �������ݼ�
        if (!load_training_set(train_db, data_path)) {
            throw std::runtime_error("ѵ��������ʧ��");
        }
        if (!load_test_set(test_db, data_path)) {
            throw std::runtime_error("���Լ�����ʧ��");
        }
        string file_config = "C:\\Users\\Lenovo\\Desktop\\AI project\\Number classification\\config.json";
        NetConfig tmp;
        read_layeronfig_from_json(file_config, tmp);
        NeuralNetwork net(tmp, &train_db, Loss_function::CRE);
        double rate = 0.005;
        for (int i = 1; i <= 100; i++)
        {
            cout << net.test(100)<<" "<<net.training(10)<<endl;
            if (i % 10 == 0)net.model_export("C:\\Users\\Lenovo\\Desktop\\AI project\\Number classification\\model.txt");
            rate *= 0.997;
            net.learning_rate(rate);
        }
        print_dataset_info(train_db, test_db);
    }
    catch (const std::exception& e) {
        std::cerr << "����: " << e.what() << std::endl;
        return -1;
    }

    // ������Դ
    train_db.clear();
    test_db.clear();
    std::cout << "\n����ִ�����" << std::endl;

    return 0;
}