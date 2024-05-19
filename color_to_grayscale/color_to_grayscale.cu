#include "../common.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void color_to_grayscale_kernel(unsigned char *input, unsigned char *output, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height)
    {
        int gray_offset = row * width + col;
        int color_offset = gray_offset * 3;
        unsigned char r = input[color_offset];
        unsigned char g = input[color_offset + 1];
        unsigned char b = input[color_offset + 2];
        output[gray_offset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

void color_to_grayscale(unsigned char *input_h, unsigned char *output_h, int width, int height)
{
    size_t grayscale_size = width * height * sizeof(unsigned char);
    size_t color_size = 3 * grayscale_size;

    // Allocate memory on the device
    unsigned char *input_d, *output_d;
    CUDA_CHECK(cudaMalloc(&input_d, color_size));
    CUDA_CHECK(cudaMalloc(&output_d, grayscale_size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(input_d, input_h, color_size, cudaMemcpyHostToDevice));

    // Launch the kernel
    dim3 block(16, 16, 1);
    dim3 grid(ceil(width / float(block.x)), ceil(height / float(block.y)), 1); // make sure to float() the ints
    color_to_grayscale_kernel<<<grid, block>>>(input_d, output_d, width, height);

    // Copy data from device to host
    CUDA_CHECK(cudaMemcpy(output_h, output_d, grayscale_size, cudaMemcpyDeviceToHost));

    // Free memory on the device
    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
}

int main()
{
    const char *image_name = "image.jpg";
    int width, height, channels;
    unsigned char *img = stbi_load(image_name, &width, &height, &channels, 0);
    if (img == NULL)
    {
        printf("Error loading the image\n");
        return 1;
    }

    if (channels != 3)
    {
        printf("The image must have 3 channels to be converted to grayscale\n");
        return 1;
    }

    printf("Loaded image with a resolution of %dx%d\n", width, height);

    unsigned char *grayscale_img = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    color_to_grayscale(img, grayscale_img, width, height);

    int quality = 100;
    if (!stbi_write_jpg("output.jpg", width, height, 1, grayscale_img, quality))
    {
        printf("Error writing the image\n");
        return 1;
    }

    stbi_image_free(img);
    free(grayscale_img);
}
