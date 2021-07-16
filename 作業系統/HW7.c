#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// process num=5,resources num=4
#define PROCESS_NUM 5
#define RESOURCE_NUM 4

int main(){
    int current[PROCESS_NUM][RESOURCE_NUM] = {};
    int max[PROCESS_NUM][RESOURCE_NUM] = {};
    int alloc[RESOURCE_NUM] = {};
    int process_i[RESOURCE_NUM], resources_max[RESOURCE_NUM], available[RESOURCE_NUM], running[PROCESS_NUM], counter = PROCESS_NUM, safe, exec, i, j, req_i; 

    printf("the number of allocated resources of every process:\n");
    for(i = 0; i < PROCESS_NUM; i++){
        printf("P%d allocation:",i);
        for(j = 0; j < RESOURCE_NUM; j++)
            scanf("%d", &current[i][j]);
    }
    printf("the number of MAX resources of every process:\n");
    for(i = 0; i < PROCESS_NUM; i++){
        printf("P%d MAX:",i);
        for(j = 0; j < RESOURCE_NUM; j++)
            scanf("%d", &max[i][j]);
    }
    printf("the total number of resources:\n");
    for(i = 0; i < RESOURCE_NUM; i++){
        scanf("%d", &resources_max[i]);
    }
    printf("the number of requested resources of process i:\n");
    printf("Pi: ");
    scanf("%d", &req_i);
    printf("the number of requested resource:\n");
    for(i = 0; i < RESOURCE_NUM; i++){
        scanf("%d", &process_i[i]);
    }
    for(i = 0; i < RESOURCE_NUM; i++){
        current[req_i][i] += process_i[i];
    }

    for(i = 0; i < RESOURCE_NUM; i++){
        alloc[i] = 0;
    }
    for(i = 0; i < PROCESS_NUM; i++){
        running[i] = 1;
        for (j = 0; j < RESOURCE_NUM; j++){
            alloc[j] += current[i][j];
        }
    }
    printf("Allocated resources:\n");
    for(i = 0; i < RESOURCE_NUM; i++){
        printf("%d ",alloc[i]);
    }
    printf("\n");

    for(i = 0; i < RESOURCE_NUM; i++){
	    available[i] = resources_max[i] - alloc[i];
	}
    printf("Available resources:\n");
    for (i = 0; i < RESOURCE_NUM; i++){
        	printf("%d ", available[i]);
    }
    printf("\n");

    while (counter != 0){
        safe = 0;
        for (i = 0; i < PROCESS_NUM; i++){
            if(running[i]){
                exec = 1;
                for(j = 0; j < RESOURCE_NUM; j++){
                    if (max[i][j] - current[i][j] > available[j]) {
                        exec = 0;
                        break;
                    }
                }
                if(exec){
                    printf("Process%d is executing\n", i + 1);
                    running[i] = 0;
                    counter--;
                    safe = 1;

                    for (j = 0; j < RESOURCE_NUM; j++){
                        available[j] += current[i][j];
                    }
                    break;
                }
            }
        }
        if (!safe) {
            printf("The processes are in unsafe state.\n");
            break;
        }else {
            printf("The process is in safe state\n");
            printf("Available:");

            for (i = 0; i < RESOURCE_NUM; i++) {
                printf("%d ", available[i]);
            }
            printf("\n\n");
        }
    }
    return 0;
}