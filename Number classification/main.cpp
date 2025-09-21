#include "DataBase.h"
#include<NeuralNetwork.h>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>

// 类型定义：用于读取文件头的32位无符号整数
using uchar = unsigned char;
using uint32 = uint32_t;

// 读取MNIST文件中的32位整数（MNIST使用大端字节序）
uint32 read_uint32(std::ifstream& file) {
    uint32 result;
    uchar bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    // 转换大端字节序到本地字节序
    result = (static_cast<uint32>(bytes[0]) << 24) |
        (static_cast<uint32>(bytes[1]) << 16) |
        (static_cast<uint32>(bytes[2]) << 8) |
        static_cast<uint32>(bytes[3]);
    return result;
}

// 加载图像文件到数据库
bool load_images(const std::string& filename, DataBase& db, int label_count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开图像文件: " << filename << std::endl;
        return false;
    }

    // 读取图像文件头
    uint32 magic = read_uint32(file);
    uint32 count = read_uint32(file);
    uint32 rows = read_uint32(file);
    uint32 cols = read_uint32(file);

    // 验证MNIST图像文件魔数
    if (magic != 2051) {
        std::cerr << "无效的图像文件魔数: " << magic << std::endl;
        return false;
    }

    // 检查图像数量与标签数量是否匹配
    if (count != static_cast<uint32>(label_count)) {
        std::cerr << "图像数量与标签数量不匹配" << std::endl;
        return false;
    }

    // 读取图像数据并添加到数据库
    const int pixel_count = rows * cols;
    std::vector<uchar> buffer(pixel_count);

    for (uint32 i = 0; i < count; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), pixel_count);

        Data_pair data;
        data.input = VectorXd(pixel_count);
        // 像素值归一化到[0, 1]
        for (int j = 0; j < pixel_count; ++j) {
            data.input[j] = static_cast<double>(buffer[j]) / 255.0;
        }

        db.push_back(data);
    }

    std::cout << "成功加载 " << count << " 张图像 (" << rows << "x" << cols << ")" << std::endl;
    return true;
}

// 加载标签文件并匹配到数据库
bool load_labels(const std::string& filename, DataBase& db) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开标签文件: " << filename << std::endl;
        return false;
    }

    // 读取标签文件头
    uint32 magic = read_uint32(file);
    uint32 count = read_uint32(file);

    // 验证MNIST标签文件魔数
    if (magic != 2049) {
        std::cerr << "无效的标签文件魔数: " << magic << std::endl;
        return false;
    }

    // 检查标签数量与数据库大小是否匹配
    if (count != static_cast<uint32>(db.size())) {
        std::cerr << "标签数量与图像数量不匹配" << std::endl;
        return false;
    }

    // 读取标签数据并设置到数据库
    for (uint32 i = 0; i < count; ++i) {
        uchar label;
        file.read(reinterpret_cast<char*>(&label), 1);

        // 使用独热编码表示标签（0-9共10个类别）
        db.Datas[i].target = VectorXd::Zero(10);
        db.Datas[i].target[label] = 1.0;
    }

    std::cout << "成功加载 " << count << " 个标签" << std::endl;
    return true;
}

// 加载训练集数据
bool load_training_set(DataBase& train_db, const std::string& base_path) {
    std::string img_path = base_path + "train-images.idx3-ubyte";
    std::string lbl_path = base_path + "train-labels.idx1-ubyte";

    // 先读取标签获取数量（用于验证图像数量）
    std::ifstream lbl_file(lbl_path, std::ios::binary);
    if (!lbl_file.is_open()) {
        std::cerr << "无法打开训练标签文件: " << lbl_path << std::endl;
        return false;
    }
    read_uint32(lbl_file); // 跳过魔数
    uint32 train_count = read_uint32(lbl_file);
    lbl_file.close();

    // 加载图像和标签
    if (!load_images(img_path, train_db, train_count)) return false;
    if (!load_labels(lbl_path, train_db)) return false;

    return true;
}

// 加载测试集数据
bool load_test_set(DataBase& test_db, const std::string& base_path) {
    std::string img_path = base_path + "t10k-images.idx3-ubyte";
    std::string lbl_path = base_path + "t10k-labels.idx1-ubyte";

    // 先读取标签获取数量
    std::ifstream lbl_file(lbl_path, std::ios::binary);
    if (!lbl_file.is_open()) {
        std::cerr << "无法打开测试标签文件: " << lbl_path << std::endl;
        return false;
    }
    read_uint32(lbl_file); // 跳过魔数
    uint32 test_count = read_uint32(lbl_file);
    lbl_file.close();

    // 加载图像和标签
    if (!load_images(img_path, test_db, test_count)) return false;
    if (!load_labels(lbl_path, test_db)) return false;

    return true;
}

// 显示数据集基本信息
void print_dataset_info(DataBase& train_db, DataBase& test_db) {
    std::cout << "\n数据集信息:" << std::endl;
    std::cout << "训练集大小: " << train_db.size() << std::endl;
    std::cout << "测试集大小: " << test_db.size() << std::endl;
    if (train_db.size() > 0) {
        std::cout << "输入维度: " << train_db.Datas[0].input.size() << std::endl;
        std::cout << "输出维度: " << train_db.Datas[0].target.size() << std::endl;
    }
}

int main() {
    // 数据集路径
    const std::string data_path = "C:\\Users\\Lenovo\\Desktop\\AI project\\Number classification\\";

    // 初始化数据库（训练集比例80%，仅对训练集数据库有效）
    DataBase train_db(0.8);
    DataBase test_db(1.0);  // 测试集不划分，全部用于测试

    try {
        // 加载数据集
        if (!load_training_set(train_db, data_path)) {
            throw std::runtime_error("训练集加载失败");
        }
        if (!load_test_set(test_db, data_path)) {
            throw std::runtime_error("测试集加载失败");
        }
        string file_config = "C:\\Users\\Lenovo\\Desktop\\AI project\\Number classification\\config.json";
        NetConfig tmp;
        read_layeronfig_from_json(file_config, tmp);
        NeuralNetwork net(tmp, &train_db, Loss_function::CRE);
        for (int i = 1; i <= 100; i++)
        {
            cout << net.test(100)<<" "<<net.training(10)<<endl;
            if (i % 10 == 0)net.model_export("C:\\Users\\Lenovo\\Desktop\\AI project\\Number classification\\model.txt");
        }
        print_dataset_info(train_db, test_db);
    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }

    // 清理资源
    train_db.clear();
    test_db.clear();
    std::cout << "\n程序执行完毕" << std::endl;

    return 0;
}