#include <filesystem>



#include "src/vector.h"
#include "src/cuda_helpers.h"
#include "src/scene.h"




int main() {


    clock_t start, stop;
    start = clock();

    Scene s(400, 225);
    s.render();


    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cout << "Computation took " << timer_seconds << " seconds.\n";





//    std::ofstream file(path);
//
//    // Output FB as Image
//    file << "P3\n" << width << " " << height << "\n255\n";
//    for (int j = height - 1; j >= 0; j--) {
//        for (int i = 0; i < width; i++) {
//            for(int k = 0; k < 3; k++){
//
//                if (int(255.99 * fb[j * width + i][k]) < 0 or int(255.99 * fb[j * width + i][k]) > 255){
//                    std::cout << "Broken thing is at " << fb[j * width + i] << ' ' << i << ' ' << j << '\n';
//                    fb[j * width + i][0] = 1;
//                    fb[j * width + i][1] = 0;
//                    fb[j * width + i][2] = 1;
//                    break;
////                    assert(0);
//                }
//
//            }
//            file << fb[j * width + i];
//        }
//    }

    std::cout << "Drew image to file\n";



    cudaDeviceReset();

    return EXIT_SUCCESS;
}
