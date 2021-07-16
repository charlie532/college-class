#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
typedef int buffer_item;
#define BUFFER_SIZE 5

int insert_item(buffer_item item);
int remove_item(buffer_item *item);

buffer_item buffer[BUFFER_SIZE];
pthread_mutex_t mutex;
sem_t full, empty;
int count, insertPointer, removePointer;

void *consumer(void *param);
void *producer(void *param);

int insert_item(buffer_item item){
	int success;
	sem_wait(&empty);
	pthread_mutex_lock(&mutex);

	if( count != BUFFER_SIZE){
		buffer[insertPointer] = item;
		insertPointer = (insertPointer + 1) % BUFFER_SIZE;
		count++;
		success = 0;
	}
	else
	success = -1;

	pthread_mutex_unlock(&mutex);
	sem_post(&full);

	return success;
}

int remove_item(buffer_item *item){
	int success;
	
	sem_wait(&full);
	pthread_mutex_lock(&mutex);
	
	if( count != 0){
		*item = buffer[removePointer];
		removePointer = (removePointer + 1) % BUFFER_SIZE;
		count--;
		success = 0;
	}
	else
		success = -1;

	pthread_mutex_unlock(&mutex);
	sem_post(&empty);
	
	return success;
}

int main(int argc, char **argv){
	if (argc != 4){
		fprintf(stderr, "Useage: <sleep time> <producer threads> <consumer threads>\n");
		exit(1);
	}

	int stime = strtol(argv[1], NULL, 0);
	int num_producer = strtol(argv[2], NULL, 0);
	int num_consumer = strtol(argv[3], NULL, 0);
	printf("%d %d %d\n", stime , num_producer, num_consumer);

	int i;
	srand(time(NULL));
	pthread_mutex_init(&mutex, NULL);
	sem_init(&empty, 0, BUFFER_SIZE);
	sem_init(&full, 0, 0);
	count = insertPointer = removePointer = 0;

	pthread_t producers[num_producer];
	pthread_t consumers[num_consumer];
	for(i = 0; i < num_producer; i++)
		pthread_create(&producers[i], NULL, producer, NULL);
	for(i = 0; i < num_consumer; i++)
		pthread_create(&consumers[i], NULL, consumer, NULL);

	sleep(stime);
	return 0;
}

void *producer(void *param){
	buffer_item item;
	while(1){
		sleep(rand() % 5 + 1);
		item = rand();
		if(insert_item(item))
			printf("Error occured\n");
		else
			printf("Producer produced %d\n", item);
	}
}


void *consumer(void *param){
	buffer_item item;
	while(1){
		sleep(rand() % 5 + 1);
		if(remove_item(&item))
			printf("Error occured\n");
		else
			printf("Consumer consumed %d\n", item);
	}
}