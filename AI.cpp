#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <cmath>  // 引入cmath用于nan检测

// 生成10x10迷宫的函数（返回double类型矩阵）
static Eigen::MatrixXd generateMaze() {
    // 初始化10x10矩阵，全部设为墙(1.0)，使用double类型
    Eigen::MatrixXd maze = Eigen::MatrixXd::Ones(10, 10);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 9);
    
    std::function<void(int, int)> carveMaze = [&](int x, int y) {
        // 标记当前位置为可走(0.0)
        maze(y, x) = 0.0;
        
        // 定义四个方向
        std::vector<std::pair<int, int>> directions;
        directions.push_back(std::make_pair(0, -1));
        directions.push_back(std::make_pair(1, 0));
        directions.push_back(std::make_pair(0, 1));
        directions.push_back(std::make_pair(-1, 0));
        std::shuffle(directions.begin(), directions.end(), gen);
        
        for (size_t i = 0; i < directions.size(); ++i) {
            int dx = directions[i].first;
            int dy = directions[i].second;
            int nx = x + 2 * dx;
            int ny = y + 2 * dy;
            
            // 检查新位置是否在范围内且为墙(1.0)
            if (nx >= 0 && nx < 10 && ny >= 0 && ny < 10 && maze(ny, nx) == 1.0) {
                // 打通中间的墙（设为可走0.0）
                maze(y + dy, x + dx) = 0.0;
                carveMaze(nx, ny);
            }
        }
    };
    
    // 随机生成起点（奇数位置）
    int startX = dist(gen);
    int startY = dist(gen);
    startX = startX % 2 == 0 ? startX + 1 : startX;
    startY = startY % 2 == 0 ? startY + 1 : startY;
    startX = std::min(startX, 9);
    startY = std::min(startY, 9);
    
    carveMaze(startX, startY);
    
    // 收集边缘位置作为终点候选
    std::vector<std::pair<int, int>> edgePositions;
    for (int i = 0; i < 10; ++i) {
        if (maze(0, i) == 0.0) edgePositions.emplace_back(i, 0);
        if (maze(9, i) == 0.0) edgePositions.emplace_back(i, 9);
        if (i > 0 && i < 9) {
            if (maze(i, 0) == 0.0) edgePositions.emplace_back(0, i);
            if (maze(i, 9) == 0.0) edgePositions.emplace_back(9, i);
        }
    }
    
    // 确保至少有一个终点候选
    if (edgePositions.empty()) {
        for (int y = 0; y < 10; ++y) {
            for (int x = 0; x < 10; ++x) {
                if (maze(y, x) == 0.0) {
                    edgePositions.emplace_back(x, y);
                    break;
                }
            }
            if (!edgePositions.empty()) break;
        }
    }
    
    // 随机选择终点并标记为-1.0
    std::uniform_int_distribution<int> edgeDist(0, edgePositions.size() - 1);
    int endX = edgePositions[edgeDist(gen)].first;
    int endY = edgePositions[edgeDist(gen)].second;
    maze(endY, endX) = -1.0;  // 终点用-1.0标记（double类型）
    
    // 检测迷宫矩阵是否包含NaN
    if (maze.hasNaN()) {
        std::cerr << "Error: Generated maze contains NaN values" << std::endl;
    }

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            if (std::isnan(maze(i, j))) {
                maze(i, j) = 1.0;  // 将NaN强制设为墙
                std::cerr << "Warning: Fixed NaN in maze at (" << i << "," << j << ")" << std::endl;
            }
        }
    }
    
    return maze;
}

// 经验结构体：迷宫使用double类型矩阵（与网络兼容）
struct experience {
    Eigen::MatrixXd maze;  // double类型矩阵
    int ax, ay;
    int x, y, nx, ny;
    double reward;
    int boom=1;
};
std::vector<experience> exper;


static void trying(int ax, int ay, int x, int y, const Eigen::MatrixXd& maze,MatrixXd Visited) {
    if (ax == -10)
        if (maze(x, y) == -1.0 || maze(x, y) == 1.0) return;  // 检查终点(-1.0)和墙(1.0)
    if (ax == -10) ax = ay = 0;
    int tmpx = x + ax, tmpy = y + ay;
    if (tmpx > 9 || tmpy > 9 || tmpx < 0 || tmpy < 0) {
        exper.push_back({maze, ax, ay, x, y, tmpx, tmpy, -1.5,0});
        return;
    }
    if(Visited(tmpx,tmpy)==1)return ;
    if (maze(tmpx, tmpy) == -1.0) {  // 检测终点（double类型）
        exper.push_back({maze, ax, ay, x, y, tmpx, tmpy, 1.0,0});
        return;
    }
    if (maze(tmpx, tmpy) == 1.0) {  // 检测墙（double类型）
        exper.push_back({maze, ax, ay, x, y, tmpx, tmpy, -1.5,0});
        return;
    }
    exper.push_back({maze, ax, ay, x, y, tmpx, tmpy, 0.02,1});
    Visited(tmpx,tmpy)=1;
    trying(0, 1, tmpx, tmpy, maze,Visited);
    trying(0, -1, tmpx, tmpy, maze,Visited);
    trying(-1, 0, tmpx, tmpy, maze,Visited);
    trying(1, 0, tmpx, tmpy, maze,Visited);
}

static void build_data(NeuralNetwork& target,double gamma,DataBase& Base)
{
    for(int i=0;i<exper.size();i++)
    {
        experience tmp=exper[i];
        VectorXd vec=tmp.maze.reshaped(),append(4);
        append<<tmp.x,tmp.y,tmp.ax,tmp.ay;
        VectorXd tar(1);
        tar[0]=tmp.reward;
        vec.conservativeResize(104);
        vec[100]=tmp.x,vec[101]=tmp.y;
        
        if(tmp.boom!=0)
        {
            vec[102]=-1,vec[103]=0;
            double q1=target.push_forward(vec)[0];
            // 检测Q值是否为NaN
            if (std::isnan(q1)) {
                std::cerr << "Error: Q1 is NaN at experience index " << i << std::endl;
                system("pause");
            }
            
            vec[102]=1,vec[103]=0;
            double q2=target.push_forward(vec)[0];
            if (std::isnan(q2)) {
                std::cerr << "Error: Q2 is NaN at experience index " << i << std::endl;
                system("pause");
            }
            
            vec[102]=0,vec[103]=1;
            double q3=target.push_forward(vec)[0];
            if (std::isnan(q3)) {
                std::cerr << "Error: Q3 is NaN at experience index " << i << std::endl;
                system("pause");
            }
            
            vec[102]=0,vec[103]=-1;
            double q4=target.push_forward(vec)[0];
            if (std::isnan(q4)) {
                std::cerr << "Error: Q4 is NaN at experience index " << i << std::endl;
                system("pause");
            }
            
            tar[0]+=gamma*max(q1,max(q2,max(q3,q4)));
            
            // 检测目标值是否为NaN
            if (std::isnan(tar[0])) {
                std::cerr << "Error: Target value is NaN at experience index " << i << std::endl;
            }
        }
        vec[102]=tmp.ax,vec[103]=tmp.ay;
        // 检测状态向量是否包含NaN
        if (vec.hasNaN()) {
            std::cerr << "Error: State vector contains NaN at experience index " << i << std::endl;
            cout<<vec[100]<<" "<<vec[101]<<endl;
            continue;  // 跳过含NaN的异常数据
        }
        Base.push_back({vec,tar});
    }
}

int main()
{
    DataBase data(0.8);
    //auto config = build_net("DQN", 0.0001, build_layers(104, 1, 4, 64));
    NetConfig config;
    read_layeronfig_from_json("C:\\Users\\Lenovo\\Desktop\\AI project\\config.json", config);
    double rate=0.0005;
    NeuralNetwork Q(config);
    Q.database(&data);
    for(int k=1;k<=10;k++)
    {
        exper.clear();
        for (int i =1;i<=200;++i)
        {
            if(exper.size()>=10000)break;
            Eigen::MatrixXd maze = generateMaze();// 生成double类型迷宫
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dist(0, 9);
            int x = dist(gen), y = dist(gen);
            trying(-10, -10, x, y, maze,MatrixXd::Zero(10,10));
        }
        for(int i=1;i<=10;i++)
        {
            data.clear();
            build_data(Q,0.8,data);
            double test_loss = Q.test(100);
            if (std::isnan(test_loss)) 
                std::cerr << "Error: Test loss is NaN at iteration " << i << ", epoch " << k << std::endl;
            cout << test_loss << endl;

            double train_loss = Q.training(100);
            if (std::isnan(train_loss))
                std::cerr << "Error: Training loss is NaN at iteration " << i << ", epoch " << k << std::endl;
        }
        rate*=0.995;
        Q.learning_rate(rate);
        Q.model_export("Q_model.txt");
        cout<<endl;
    }
    
}
