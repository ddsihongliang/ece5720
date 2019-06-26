/********************************************************************
    render.c is responsible for rendering the bodies' positions and 
    velocities to an ppm image
********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "string.h"

#define WIDTH 1024
#define HEIGHT 1024

typedef struct
{
    int x, y;
    int red, green, blue;
} PPMPixel;

void body_to_image(int *hdImage, PPMPixel image)
{
  hdImage[image.y * HEIGHT + image.x] = image.red;
  hdImage[image.y * HEIGHT + image.x + 1] = image.green;
  hdImage[image.y * HEIGHT + image.x + 2] = image.blue;
}

void render_bodies(const float *position, float *velocity, int number_of_bodies, int *hdImage)
{
  for (int i = 0; i < number_of_bodies; i++) {
    PPMPixel body;
    float current_x = position[i * 3];
    float current_y = position[i * 3 + 1];
    float current_z = position[i * 3 + 2];
    body.x = (int) current_x + 500;
    body.y = (int) current_y + 500;
    body.red = 255;
    body.green = 255;
    body.blue = 255;
    body_to_image(hdImage,body);
  }
}

char* itoa(int x)
{
  size_t length = snprintf( NULL, 0, "%d", x );
  char* str = malloc( length + 1 );
  snprintf( str, length + 1, "%d", x );
  return str;
}

void write_to_file(const int *hdImage, int step)
{
  FILE *fp;
  char *file_path = strcat(itoa(step), ".ppm");
  fp = fopen(file_path, "wb+");
  char *width = itoa(WIDTH);
  char *height = itoa(HEIGHT);
  fprintf(fp, "P6\n");
  fprintf(fp, width);
  fprintf(fp, " ");
  fprintf(fp, height);
  fprintf(fp, "\n");
  fprintf(fp, "255\n");
  free(width);
  free(height);
  char* data = malloc(sizeof(char) * WIDTH * HEIGHT * 3);
  for (int i = 0; i < WIDTH * HEIGHT * 3; i++) {
    data[i] = (char) hdImage[i];
  }

  fwrite(data, sizeof(short), WIDTH * HEIGHT * 3, fp);
  free(data);
  fclose(fp);
}

int* initialize_image()
{
  int *image = malloc(sizeof(int) * WIDTH * HEIGHT * 3);
  for (int i = 0; i < WIDTH * HEIGHT; i++) {
    image[i * 3] = 0;
    image[i * 3 + 1] = 0;
    image[i * 3 + 2] = 0;
  }
  return image;
}

void create_frame(float *position, float *velocity, int num_of_bodies, int step)
{
    int *hdImage = initialize_image();

    printf("Rendering Pardicles...\n");

    render_bodies(position, velocity, num_of_bodies, hdImage);
    write_to_file(hdImage, step);

    printf("Successfully rendered to the file\n");
    free(hdImage);
}

int main()
{
  float *pos = malloc(sizeof(float));
  float *vel = malloc(sizeof(float));
  create_frame(pos, vel, 1, 1);
}