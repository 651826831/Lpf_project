#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>    
#include <unistd.h>   
#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int  verbose = 1;
int count_warp_level = 1;

__managed__ uint32_t print_level = 4;  

__managed__ uint32_t block_num = 0;
__managed__ uint32_t thrd_num = 0;

/* record the current register status: 
 * 3-> This register has not been store or load yet. 
 * 0-> Stored with REG, havent been loaded.
 * 1-> Stored with  REG, loaded by REG.
 * 2-> Store by MERF, loaded by MERF.
 * 4-> Store by REG, loade by MERF / Store by MERF, loaded by REG.
 * 5-> Store by MERF, havebt been loaded.
 */
__managed__ uint32_t reg_status[256][256][256];
__managed__ uint32_t mov_instr_locate_index[256][256][256];
__managed__ uint32_t counter[256][256];
__managed__ uint64_t evecounter = 0;
uint32_t ker_begin_interval = 0;
uint32_t ker_end_interval = UINT32_MAX;
uint32_t kernel_id = 0;
pthread_mutex_t mutex;
uint64_t tot_app_instrs = 0;

/* delete_list is used to record dead store instrs 
 * 3-> Initial value;
 * 0-> The intstr is a dead store;
 */
__managed__ uint32_t delete_list[256][256][256]; 

typedef struct {
	int cta_id_x;
	int cta_id_y;
	int cta_id_z;
	int warp_id;
	int opcode_id;
	uint64_t addrs[32];
} mem_access_t;

extern "C" __device__ __noinline__ int six_op_instr_process(int pred,
              					     int op0_reg_dst_num,
						     int op0_type,
						     int op1_reg_dst_num,
						     int op1_type,
						     int op2_reg_dst_num,
						     int op2_type,
						     int op3_reg_dst_num,
						     int op3_type,
						     int op4_reg_dst_num,
						     int op4_type,
						     int op5_reg_dst_num,
						     int op5_type
						     ) {
	if (!pred) {
		return 0;
	}


	/* all the active threads will compute the active mask */
	const int active_mask = __ballot(1);
	/* compute the predicate mask */
	const int predicate_mask = __ballot(pred);
	/* each thread will get a lane id (get_lane_id is in utils/utils.h) */
	const int laneid = get_laneid();
	/* get the id of the first active thread */
	const int first_laneid = __ffs(active_mask) - 1;
	/* count all the active thread */
	const int num_threads = __popc(predicate_mask);

	mem_access_t ma;
	int4 cta = get_ctaid();
	ma.cta_id_x = cta.x;
	ma.cta_id_y = cta.y;
	ma.cta_id_z = cta.z;
	int block_id = cta.x + cta.x * cta.y + cta.x * cta.y * cta.z; 
	thrd_num = num_threads;
	counter[block_id][laneid] = counter[block_id][laneid] + 1;

	if (op1_type == 1 || op1_type == 2) {
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 0)
				reg_status[block_id][laneid][op1_reg_dst_num] = 1;
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 2)
				reg_status[block_id][laneid][op1_reg_dst_num] = 4;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 0)
				reg_status[block_id][laneid][op1_reg_dst_num] = 2;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 1)
				reg_status[block_id][laneid][op1_reg_dst_num] = 4;
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 5)
				reg_status[block_id][laneid][op1_reg_dst_num] = 1;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 5)
				reg_status[block_id][laneid][op1_reg_dst_num] = 2;
	}
	if (op2_type == 1 || op2_type == 2) {
			if(op2_type == 1 && reg_status[block_id][laneid][op2_reg_dst_num] == 0)
				reg_status[block_id][laneid][op2_reg_dst_num] = 1;
			if(op2_type == 1 && reg_status[block_id][laneid][op2_reg_dst_num] == 2)
				reg_status[block_id][laneid][op2_reg_dst_num] = 4;
			if(op2_type == 2 && reg_status[block_id][laneid][op2_reg_dst_num] == 0)
				reg_status[block_id][laneid][op2_reg_dst_num] = 2;
			if(op2_type == 2 && reg_status[block_id][laneid][op2_reg_dst_num] == 1)
				reg_status[block_id][laneid][op2_reg_dst_num] = 4;
			if(op2_type == 1 && reg_status[block_id][laneid][op2_reg_dst_num] == 5)
				reg_status[block_id][laneid][op2_reg_dst_num] = 1;
			if(op2_type == 2 && reg_status[block_id][laneid][op2_reg_dst_num] == 5)
				reg_status[block_id][laneid][op2_reg_dst_num] = 2;
	}
	if (op3_type == 1 || op3_type == 2) {
			if(op3_type == 1 && reg_status[block_id][laneid][op3_reg_dst_num] == 0)
				reg_status[block_id][laneid][op3_reg_dst_num] = 1;
			if(op3_type == 1 && reg_status[block_id][laneid][op3_reg_dst_num] == 2)
				reg_status[block_id][laneid][op3_reg_dst_num] = 4;
			if(op3_type == 2 && reg_status[block_id][laneid][op3_reg_dst_num] == 0)
				reg_status[block_id][laneid][op3_reg_dst_num] = 2;
			if(op3_type == 2 && reg_status[block_id][laneid][op3_reg_dst_num] == 1)
				reg_status[block_id][laneid][op3_reg_dst_num] = 4;
			if(op3_type == 1 && reg_status[block_id][laneid][op3_reg_dst_num] == 5)
				reg_status[block_id][laneid][op3_reg_dst_num] = 1;
			if(op3_type == 2 && reg_status[block_id][laneid][op3_reg_dst_num] == 5)
				reg_status[block_id][laneid][op3_reg_dst_num] = 2;
	}
	if (op4_type == 1 || op4_type == 2) {
			if(op4_type == 1 && reg_status[block_id][laneid][op4_reg_dst_num] == 0)
				reg_status[block_id][laneid][op4_reg_dst_num] = 1;
			if(op4_type == 1 && reg_status[block_id][laneid][op4_reg_dst_num] == 2)
				reg_status[block_id][laneid][op4_reg_dst_num] = 4;
			if(op4_type == 2 && reg_status[block_id][laneid][op4_reg_dst_num] == 0)
				reg_status[block_id][laneid][op4_reg_dst_num] = 2;
			if(op4_type == 2 && reg_status[block_id][laneid][op4_reg_dst_num] == 1)
				reg_status[block_id][laneid][op4_reg_dst_num] = 4;
			if(op4_type == 1 && reg_status[block_id][laneid][op4_reg_dst_num] == 5)
				reg_status[block_id][laneid][op4_reg_dst_num] = 1;
			if(op4_type == 2 && reg_status[block_id][laneid][op4_reg_dst_num] == 5)
				reg_status[block_id][laneid][op4_reg_dst_num] = 2;
	}
	if (op5_type == 1 || op5_type == 2) {
			if(op5_type == 1 && reg_status[block_id][laneid][op5_reg_dst_num] == 0)
				reg_status[block_id][laneid][op5_reg_dst_num] = 1;
			if(op5_type == 1 && reg_status[block_id][laneid][op5_reg_dst_num] == 2)
				reg_status[block_id][laneid][op5_reg_dst_num] = 4;
			if(op5_type == 2 && reg_status[block_id][laneid][op5_reg_dst_num] == 0)
				reg_status[block_id][laneid][op5_reg_dst_num] = 2;
			if(op5_type == 2 && reg_status[block_id][laneid][op5_reg_dst_num] == 1)
				reg_status[block_id][laneid][op5_reg_dst_num] = 4;
			if(op5_type == 1 && reg_status[block_id][laneid][op5_reg_dst_num] == 5)
				reg_status[block_id][laneid][op5_reg_dst_num] = 1;
			if(op5_type == 2 && reg_status[block_id][laneid][op5_reg_dst_num] == 5)
				reg_status[block_id][laneid][op5_reg_dst_num] = 2;
	}
	if (op0_type == 1 || op0_type == 2) {
		if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 0) {
			delete_list[block_id][laneid][mov_instr_locate_index
				[block_id][laneid][op0_reg_dst_num]] = 0;
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] =
			       	counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 0;
		}
		if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 5) {
			delete_list[block_id][laneid][mov_instr_locate_index
				[block_id][laneid][op0_reg_dst_num]] = 0;
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 5;
		}
		if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 5) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 0;
		}
		if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 0) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] =
			       	counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 5;
		}
		if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 3) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 0;
		}
		if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 3) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 5;
		}
		if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 1) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 0;
		}
		if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 4) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 0;
		}
		if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 2) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 5;
		}
		if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 4) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 5;
		}
	}

	if(print_level == 4 && laneid == 0 && block_id == 0){
		
		// printf("This is a no_op instr\n");
		for(int j=0; j<20; j++){
			printf("%d", delete_list[0][0][j]);
			if(j!=19)
				printf(",");
		}
		printf("\n");
	}
	return 0;
}
NVBIT_EXPORT_FUNC(six_op_instr_process);

extern "C" __device__ __noinline__ int five_op_instr_process(int pred,
              					     int op0_reg_dst_num,
						     int op0_type,
						     int op1_reg_dst_num,
						     int op1_type,
						     int op2_reg_dst_num,
						     int op2_type,
						     int op3_reg_dst_num,
						     int op3_type,
						     int op4_reg_dst_num,
						     int op4_type
						     ) {
	if (!pred) {
		return 0;
	}
	unsigned int active_mask = __ballot(1);
	const int predicate_mask = __ballot(pred);
	const int laneid = get_laneid();
	const int first_laneid = __ffs(active_mask) - 1;
	const int num_threads = __popc(predicate_mask);
	mem_access_t ma;
	int4 cta = get_ctaid();
	ma.cta_id_x = cta.x;
	ma.cta_id_y = cta.y;
	ma.cta_id_z = cta.z;
	int block_id = cta.x + cta.x * cta.y + cta.x * cta.y * cta.z; 
	thrd_num = num_threads;
	
	counter[block_id][laneid] = counter[block_id][laneid] + 1;

	if (op1_type == 1 || op1_type == 2) {
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 0)
				reg_status[block_id][laneid][op1_reg_dst_num] = 1;
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 2)
				reg_status[block_id][laneid][op1_reg_dst_num] = 4;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 0)
				reg_status[block_id][laneid][op1_reg_dst_num] = 2;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 1)
				reg_status[block_id][laneid][op1_reg_dst_num] = 4;
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 5)
				reg_status[block_id][laneid][op1_reg_dst_num] = 1;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 5)
				reg_status[block_id][laneid][op1_reg_dst_num] = 2;
	}
	if (op2_type == 1 || op2_type == 2) {
			if(op2_type == 1 && reg_status[block_id][laneid][op2_reg_dst_num] == 0)
				reg_status[block_id][laneid][op2_reg_dst_num] = 1;
			if(op2_type == 1 && reg_status[block_id][laneid][op2_reg_dst_num] == 2)
				reg_status[block_id][laneid][op2_reg_dst_num] = 4;
			if(op2_type == 2 && reg_status[block_id][laneid][op2_reg_dst_num] == 0)
				reg_status[block_id][laneid][op2_reg_dst_num] = 2;
			if(op2_type == 2 && reg_status[block_id][laneid][op2_reg_dst_num] == 1)
				reg_status[block_id][laneid][op2_reg_dst_num] = 4;
			if(op2_type == 1 && reg_status[block_id][laneid][op2_reg_dst_num] == 5)
				reg_status[block_id][laneid][op2_reg_dst_num] = 1;
			if(op2_type == 2 && reg_status[block_id][laneid][op2_reg_dst_num] == 5)
				reg_status[block_id][laneid][op2_reg_dst_num] = 2;
	}
	if (op3_type == 1 || op3_type == 2) {
			if(op3_type == 1 && reg_status[block_id][laneid][op3_reg_dst_num] == 0)
				reg_status[block_id][laneid][op3_reg_dst_num] = 1;
			if(op3_type == 1 && reg_status[block_id][laneid][op3_reg_dst_num] == 2)
				reg_status[block_id][laneid][op3_reg_dst_num] = 4;
			if(op3_type == 2 && reg_status[block_id][laneid][op3_reg_dst_num] == 0)
				reg_status[block_id][laneid][op3_reg_dst_num] = 2;
			if(op3_type == 2 && reg_status[block_id][laneid][op3_reg_dst_num] == 1)
				reg_status[block_id][laneid][op3_reg_dst_num] = 4;
			if(op3_type == 1 && reg_status[block_id][laneid][op3_reg_dst_num] == 5)
				reg_status[block_id][laneid][op3_reg_dst_num] = 1;
			if(op3_type == 2 && reg_status[block_id][laneid][op3_reg_dst_num] == 5)
				reg_status[block_id][laneid][op3_reg_dst_num] = 2;
	}
	if (op4_type == 1 || op4_type == 2) {
			if(op4_type == 1 && reg_status[block_id][laneid][op4_reg_dst_num] == 0)
				reg_status[block_id][laneid][op4_reg_dst_num] = 1;
			if(op4_type == 1 && reg_status[block_id][laneid][op4_reg_dst_num] == 2)
				reg_status[block_id][laneid][op4_reg_dst_num] = 4;
			if(op4_type == 2 && reg_status[block_id][laneid][op4_reg_dst_num] == 0)
				reg_status[block_id][laneid][op4_reg_dst_num] = 2;
			if(op4_type == 2 && reg_status[block_id][laneid][op4_reg_dst_num] == 1)
				reg_status[block_id][laneid][op4_reg_dst_num] = 4;
			if(op4_type == 1 && reg_status[block_id][laneid][op4_reg_dst_num] == 5)
				reg_status[block_id][laneid][op4_reg_dst_num] = 1;
			if(op4_type == 2 && reg_status[block_id][laneid][op4_reg_dst_num] == 5)
				reg_status[block_id][laneid][op4_reg_dst_num] = 2;
	}

	if (op0_type == 1 || op0_type == 2) {
		if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 0) {
			delete_list[block_id][laneid][mov_instr_locate_index
				[block_id][laneid][op0_reg_dst_num]] = 0;
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 0;
		}
		if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 5) {
			delete_list[block_id][laneid][mov_instr_locate_index
				[block_id][laneid][op0_reg_dst_num]] = 0;
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 5;
		}
		if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 5) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 0;
		}
		if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 0) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 5;
		}
		if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 3) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 0;
		}
		if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 3) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 5;
		}
		if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 1) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 0;
		}
		if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 4) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 0;
		}
		if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 2) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 5;
		}
		if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 4) {
			mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = 
				counter[block_id][laneid];
			reg_status[block_id][laneid][op0_reg_dst_num] = 5;
		}
	}
	
	if(print_level == 4 && laneid == 0 && block_id == 0 ){
		for(int j=0; j<20; j++){
			printf("%d", delete_list[0][0][j]);
			if(j!=19)
				printf(",");
		}
		printf("\n");
	}
	
	return 0;
}
NVBIT_EXPORT_FUNC(five_op_instr_process);

extern "C" __device__ __noinline__ int four_op_instr_process(int pred,
              					     int op0_reg_dst_num,
						     int op0_type,
						     int op1_reg_dst_num,
						     int op1_type,
						     int op2_reg_dst_num,
						     int op2_type,
						     int op3_reg_dst_num,
						     int op3_type
						     ) {
	if (!pred) {
		return 0;
	}

	unsigned int active_mask = __ballot(1);
	const int predicate_mask = __ballot(pred);
	const int laneid = get_laneid();
	const int first_laneid = __ffs(active_mask) - 1;
	const int num_threads = __popc(predicate_mask);
	
	mem_access_t ma;
	int4 cta = get_ctaid();
	ma.cta_id_x = cta.x;
	ma.cta_id_y = cta.y;
	ma.cta_id_z = cta.z;
	thrd_num = num_threads;
	
	int block_id = cta.x + cta.x * cta.y + cta.x * cta.y * cta.z; 
	counter[block_id][laneid] = counter[block_id][laneid] + 1;
	
	if (op1_type == 1 || op1_type == 2) {
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 0)
				reg_status[block_id][laneid][op1_reg_dst_num] = 1;
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 2)
				reg_status[block_id][laneid][op1_reg_dst_num] = 4;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 0)
				reg_status[block_id][laneid][op1_reg_dst_num] = 2;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 1)
				reg_status[block_id][laneid][op1_reg_dst_num] = 4;
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 5)
				reg_status[block_id][laneid][op1_reg_dst_num] = 1;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 5)
				reg_status[block_id][laneid][op1_reg_dst_num] = 2;
	}
	if (op2_type == 1 || op2_type == 2) {
			if(op2_type == 1 && reg_status[block_id][laneid][op2_reg_dst_num] == 0)
				reg_status[block_id][laneid][op2_reg_dst_num] = 1;
			if(op2_type == 1 && reg_status[block_id][laneid][op2_reg_dst_num] == 2)
				reg_status[block_id][laneid][op2_reg_dst_num] = 4;
			if(op2_type == 2 && reg_status[block_id][laneid][op2_reg_dst_num] == 0)
				reg_status[block_id][laneid][op2_reg_dst_num] = 2;
			if(op2_type == 2 && reg_status[block_id][laneid][op2_reg_dst_num] == 1)
				reg_status[block_id][laneid][op2_reg_dst_num] = 4;
			if(op2_type == 1 && reg_status[block_id][laneid][op2_reg_dst_num] == 5)
				reg_status[block_id][laneid][op2_reg_dst_num] = 1;
			if(op2_type == 2 && reg_status[block_id][laneid][op2_reg_dst_num] == 5)
				reg_status[block_id][laneid][op2_reg_dst_num] = 2;
	}

	if (op3_type == 1 || op3_type == 2) {
			if(op3_type == 1 && reg_status[block_id][laneid][op3_reg_dst_num] == 0)
				reg_status[block_id][laneid][op3_reg_dst_num] = 1;
			if(op3_type == 1 && reg_status[block_id][laneid][op3_reg_dst_num] == 2)
				reg_status[block_id][laneid][op3_reg_dst_num] = 4;
			if(op3_type == 2 && reg_status[block_id][laneid][op3_reg_dst_num] == 0)
				reg_status[block_id][laneid][op3_reg_dst_num] = 2;
			if(op3_type == 2 && reg_status[block_id][laneid][op3_reg_dst_num] == 1)
				reg_status[block_id][laneid][op3_reg_dst_num] = 4;
			if(op3_type == 1 && reg_status[block_id][laneid][op3_reg_dst_num] == 5)
				reg_status[block_id][laneid][op3_reg_dst_num] = 1;
			if(op3_type == 2 && reg_status[block_id][laneid][op3_reg_dst_num] == 5)
				reg_status[block_id][laneid][op3_reg_dst_num] = 2;
	}

	if (op0_type == 1 || op0_type == 2) {
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 0) {
				delete_list[block_id][laneid][mov_instr_locate_index[block_id][laneid][op0_reg_dst_num]] = 0;
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 5) {
				delete_list[block_id][laneid][mov_instr_locate_index[block_id][laneid][op0_reg_dst_num]] = 0;
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 5) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 0) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 3) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 3) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 1) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 4) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 2) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 4) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
	}

	if(print_level == 4 && laneid == 0 && block_id == 0){
		
		// printf("This is a no_op instr\n");
		for(int j=0; j<20; j++){
			printf("%d", delete_list[0][0][j]);
			if(j!=19)
				printf(",");
		}
		printf("\n");
	}
	
	return 0;
}
NVBIT_EXPORT_FUNC(four_op_instr_process);

extern "C" __device__ __noinline__ int three_op_instr_process(int pred,
              					     int op0_reg_dst_num,
						     int op0_type,
						     int op1_reg_dst_num,
						     int op1_type,
						     int op2_reg_dst_num,
						     int op2_type
						     ) {
	if (!pred) {
		return 0;
	}
	const int active_mask = __ballot(1);
	const int predicate_mask = __ballot(pred);
	const int laneid = get_laneid();
	const int first_laneid = __ffs(active_mask) - 1;
	const int num_threads = __popc(predicate_mask);

	mem_access_t ma;
	int4 cta = get_ctaid();
	ma.cta_id_x = cta.x;
	ma.cta_id_y = cta.y;
	ma.cta_id_z = cta.z;
	int block_id = cta.x + cta.x * cta.y + cta.x * cta.y * cta.z; 
	thrd_num = num_threads;

	counter[block_id][laneid] = counter[block_id][laneid] + 1;
	
	if (op1_type == 1 || op1_type == 2) {
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 0)
				reg_status[block_id][laneid][op1_reg_dst_num] = 1;
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 2)
				reg_status[block_id][laneid][op1_reg_dst_num] = 4;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 0)
				reg_status[block_id][laneid][op1_reg_dst_num] = 2;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 1)
				reg_status[block_id][laneid][op1_reg_dst_num] = 4;
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 5)
				reg_status[block_id][laneid][op1_reg_dst_num] = 1;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 5)
				reg_status[block_id][laneid][op1_reg_dst_num] = 2;
	}
	if (op2_type == 1 || op2_type == 2) {
			if(op2_type == 1 && reg_status[block_id][laneid][op2_reg_dst_num] == 0)
				reg_status[block_id][laneid][op2_reg_dst_num] = 1;
			if(op2_type == 1 && reg_status[block_id][laneid][op2_reg_dst_num] == 2)
				reg_status[block_id][laneid][op2_reg_dst_num] = 4;
			if(op2_type == 2 && reg_status[block_id][laneid][op2_reg_dst_num] == 0)
				reg_status[block_id][laneid][op2_reg_dst_num] = 2;
			if(op2_type == 2 && reg_status[block_id][laneid][op2_reg_dst_num] == 1)
				reg_status[block_id][laneid][op2_reg_dst_num] = 4;
			if(op2_type == 1 && reg_status[block_id][laneid][op2_reg_dst_num] == 5)
				reg_status[block_id][laneid][op2_reg_dst_num] = 1;
			if(op2_type == 2 && reg_status[block_id][laneid][op2_reg_dst_num] == 5)
				reg_status[block_id][laneid][op2_reg_dst_num] = 2;
	}
	if (op0_type == 1 || op0_type == 2) {
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 0) {
				delete_list[block_id][laneid][mov_instr_locate_index[block_id][laneid][op0_reg_dst_num]] = 0;
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 5) {
				delete_list[block_id][laneid][mov_instr_locate_index[block_id][laneid][op0_reg_dst_num]] = 0;
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 5) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 0) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 3) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 3) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 1) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 4) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 2) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 4) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
	}
	
	if(print_level == 4 && laneid == 0 && block_id == 0){
		
		// printf("This is a no_op instr\n");
		for(int j=0; j<20; j++){
			printf("%d", delete_list[0][0][j]);
			if(j!=19)
				printf(",");
		}
		printf("\n");
	}
	return 0;
}
NVBIT_EXPORT_FUNC(three_op_instr_process);

extern "C" __device__ __noinline__ int two_op_instr_process(int pred,
              					     int op0_reg_dst_num,
						     int op0_type,
						     int op1_reg_dst_num,
						     int op1_type
						     ) {
	if (!pred) {
		return 0;
	}
	/* all the active threads will compute the active mask */
	const int active_mask = __ballot(1);
	/* compute the predicate mask */
	const int predicate_mask = __ballot(pred);
	/* each thread will get a lane id (get_lane_id is in utils/utils.h) */
	const int laneid = get_laneid();
	/* get the id of the first active thread */
	const int first_laneid = __ffs(active_mask) - 1;
	/* count all the active thread */
	const int num_threads = __popc(predicate_mask);

	mem_access_t ma;
	int4 cta = get_ctaid();
	ma.cta_id_x = cta.x;
	ma.cta_id_y = cta.y;
	ma.cta_id_z = cta.z;
	int block_id = cta.x + cta.x * cta.y + cta.x * cta.y * cta.z; 
	thrd_num = num_threads;

	if(block_num < block_id)
		block_num = block_id;


	counter[block_id][laneid] = counter[block_id][laneid] + 1;
	
	if (op1_type == 1 || op1_type == 2) {
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 0)
				reg_status[block_id][laneid][op1_reg_dst_num] = 1;
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 2)
				reg_status[block_id][laneid][op1_reg_dst_num] = 4;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 0)
				reg_status[block_id][laneid][op1_reg_dst_num] = 2;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 1)
				reg_status[block_id][laneid][op1_reg_dst_num] = 4;
			if(op1_type == 1 && reg_status[block_id][laneid][op1_reg_dst_num] == 5)
				reg_status[block_id][laneid][op1_reg_dst_num] = 1;
			if(op1_type == 2 && reg_status[block_id][laneid][op1_reg_dst_num] == 5)
				reg_status[block_id][laneid][op1_reg_dst_num] = 2;
	}
	if (op0_type == 1 || op0_type == 2) {
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 0) {
				delete_list[block_id][laneid][mov_instr_locate_index[block_id][laneid][op0_reg_dst_num]] = 0;
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 5) {
				delete_list[block_id][laneid][mov_instr_locate_index[block_id][laneid][op0_reg_dst_num]] = 0;
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 5) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 0) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 3) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 3) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 1) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 1 && reg_status[block_id][laneid][op0_reg_dst_num] == 4) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 0;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 2) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
			if(op0_type == 2 && reg_status[block_id][laneid][op0_reg_dst_num] == 4) {
				mov_instr_locate_index[block_id][laneid][op0_reg_dst_num] = counter[block_id][laneid];
				reg_status[block_id][laneid][op0_reg_dst_num] = 5;
			}
	}
	
	if(print_level == 4 && laneid == 0 && block_id == 0 ){
		
		// printf("This is a no_op instr\n");
		for(int j=0; j<20; j++){
			printf("%d", delete_list[0][0][j]);
			if(j!=19)
				printf(",");
		}
		printf("\n");
	}
	return 0;
}
NVBIT_EXPORT_FUNC(two_op_instr_process);

extern "C" __device__ __noinline__ int one_op_instr_process(int pred,
              					     int op0_reg_dst_num,
						     int op0_type
						     ) {
	if (!pred) {
		return 0;
	}
	const int active_mask = __ballot(1);
	const int predicate_mask = __ballot(pred);
	const int laneid = get_laneid();
	const int first_laneid = __ffs(active_mask) - 1;
	const int num_threads = __popc(predicate_mask);
	int4 cta = get_ctaid();
	mem_access_t ma;
	ma.cta_id_x = cta.x;
	ma.cta_id_y = cta.y;
	ma.cta_id_z = cta.z;
	thrd_num = num_threads;
	
	int block_id = cta.x + cta.x * cta.y + cta.x * cta.y * cta.z; 
	counter[block_id][laneid] = counter[block_id][laneid] + 1;
	
	if(print_level == 4 && laneid == 0 && block_id == 0 ){
		
		// printf("This is a no_op instr\n");
		for(int j=0; j<20; j++){
			printf("%d", delete_list[0][0][j]);
			if(j!=19)
				printf(",");
		}
		printf("\n");
	}
	
	return 0;
}
NVBIT_EXPORT_FUNC(one_op_instr_process);


extern "C" __device__ __noinline__ int no_op_instr_process(int pred
						     ) {
	if (!pred) {
		return 0;
	}
	const int active_mask = __ballot(1);
	const int predicate_mask = __ballot(pred);
	const int laneid = get_laneid();
	const int first_laneid = __ffs(active_mask) - 1;
	const int num_threads = __popc(predicate_mask);

	mem_access_t ma;
	int4 cta = get_ctaid();
	ma.cta_id_x = cta.x;
	ma.cta_id_y = cta.y;
	ma.cta_id_z = cta.z;
	int block_id = cta.x + cta.x * cta.y + cta.x * cta.y * cta.z; 
	thrd_num = num_threads;
	
	counter[block_id][laneid] = counter[block_id][laneid] + 1;
	
	if(print_level == 4 && laneid == 0 && block_id == 0 ){
		
		// printf("This is a no_op instr\n");
		for(int j=0; j<20; j++){
			printf("%d", delete_list[0][0][j]);
			if(j!=19)
				printf(",");
		}
		printf("\n");
	}
	
	return 0;
}
NVBIT_EXPORT_FUNC(no_op_instr_process);


void nvbit_at_init() {

	GET_VAR_INT(
		instr_begin_interval, "INSTR_BEGIN", 0,
			"Beginning of the instruction interval where to apply instrumentation");
	GET_VAR_INT(
		instr_end_interval, "INSTR_END", UINT32_MAX,
		"End of the instruction interval where to apply instrumentation");
	 
	GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 1,
			                "Count warp level or thread level instructions");

	GET_VAR_INT(verbose, "TOOL_VERBOSE", 1, "Enable verbosity inside the tool");
	
	std::string pad(100, '-');
	printf("%s\n", pad.c_str());
}

void nvbit_at_term() {
    
    if(print_level == 4 && verbose == 2){
	    for(int i=0; i<block_num; i++){
		for(int j=0; j<thrd_num; j++){
			for(int k=0; k<20; k++){
				printf("[1m[40;36m%d[0m[0m", delete_list[i][j][k]);
			}
			printf("\n");
		}
		printf("\n");
	    }
    }
}	

void nvbit_at_function_first_load(CUcontext ctx, CUfunction func) {
    
    // Order test ...
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, func);
    
    struct timeval timeStart, timeEnd; 
    double runTime=0; 
    gettimeofday(&timeStart, NULL);

    for(int i = 0; i < 256; i++){
    	for(int j = 0; j < 256; j++){
    		for(int k = 0; k < 256; k++){
			reg_status[i][j][k] = 3;
    			delete_list[i][j][k] = 3;
		}
    	}
    }

    memset(counter, 0, sizeof(counter));

    /*
    const CFG_t &cfg = nvbit_get_CFG(ctx, func);
    if (cfg.is_degenerate) {
	    printf("Warning: Function %s is degenerated, we can't compute basic "
			"blocks statically",
			nvbit_get_func_name(ctx, func));
    }

    int cnt = 0;
    for (auto &bb : cfg.bbs) {
	    for (auto &i : bb->instrs) {
		    i->print(" ");
	    }
    }
    */


    // dead stroe check 
    for (auto instr : instrs) {
        
	/* Check if the instruction falls in the interval where we want to
         * instrument */
        if (instr->getIdx() < instr_begin_interval ||
            instr->getIdx() >= instr_end_interval) {
            continue;
        }

	if (verbose == 1 || verbose == 2) {
		instr->print();
	}

	int opera_number = instr->getNumOperands();
	
	if(opera_number == 0) {
		nvbit_insert_call(instr, "no_op_instr_process", IPOINT_BEFORE);
		nvbit_add_call_arg_pred_val(instr);
	}
	if(opera_number == 1) {
		const Instr::operand_t *op0 = instr->getOperand(0);
		nvbit_insert_call(instr, "one_op_instr_process", IPOINT_BEFORE);
		nvbit_add_call_arg_pred_val(instr);
		if (op0->type == Instr::REG || op0->type == Instr::MREF) {
			int Op1_num = (int)op0->value[0];
			nvbit_add_call_arg_const_val32(instr, Op1_num);
			if(op0->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op0->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op0->type != Instr::REG && op0->type != Instr::MREF) {
			int Op1_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op1_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	}

	if(opera_number == 2) {
		const Instr::operand_t *op0 = instr->getOperand(0);
		const Instr::operand_t *op1 = instr->getOperand(1);
		nvbit_insert_call(instr, "two_op_instr_process", IPOINT_BEFORE);
		nvbit_add_call_arg_pred_val(instr);
	 	if (op0->type == Instr::REG || op0->type == Instr::MREF) {
			int Op1_num = (int)op0->value[0];
			nvbit_add_call_arg_const_val32(instr, Op1_num);
			if(op0->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op0->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op0->type != Instr::REG && op0->type != Instr::MREF) {
			int Op1_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op1_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op1->type == Instr::REG || op1->type == Instr::MREF) {
			int Op2_num = (int)op1->value[0];
			nvbit_add_call_arg_const_val32(instr, Op2_num);
			if(op1->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op1->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op1->type != Instr::REG && op1->type != Instr::MREF) {
			int Op2_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op2_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	}

	if(opera_number == 3) {
		const Instr::operand_t *op0 = instr->getOperand(0);
		const Instr::operand_t *op1 = instr->getOperand(1);
		const Instr::operand_t *op2 = instr->getOperand(2);
		nvbit_insert_call(instr, "three_op_instr_process", IPOINT_BEFORE);
		nvbit_add_call_arg_pred_val(instr);
	 	if (op0->type == Instr::REG || op0->type == Instr::MREF) {
			int Op1_num = (int)op0->value[0];
			nvbit_add_call_arg_const_val32(instr, Op1_num);
			if(op0->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op0->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op0->type != Instr::REG && op0->type != Instr::MREF) {
			int Op1_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op1_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op1->type == Instr::REG || op1->type == Instr::MREF) {
			int Op2_num = (int)op1->value[0];
			nvbit_add_call_arg_const_val32(instr, Op2_num);
			if(op1->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op1->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op1->type != Instr::REG && op1->type != Instr::MREF) {
			int Op2_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op2_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op2->type == Instr::REG || op2->type == Instr::MREF) {
			
			int Op3_num = (int)op2->value[0];
			nvbit_add_call_arg_const_val32(instr, Op3_num);
			if(op2->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op2->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op2->type != Instr::REG && op2->type != Instr::MREF) {
			int Op3_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op3_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	}

	if(opera_number == 4) {
		const Instr::operand_t *op0 = instr->getOperand(0);
		const Instr::operand_t *op1 = instr->getOperand(1);
		const Instr::operand_t *op2 = instr->getOperand(2);
		const Instr::operand_t *op3 = instr->getOperand(3);
		nvbit_insert_call(instr, "four_op_instr_process", IPOINT_BEFORE);
		nvbit_add_call_arg_pred_val(instr);
	 	if (op0->type == Instr::REG || op0->type == Instr::MREF) {
			int Op1_num = (int)op0->value[0];
			nvbit_add_call_arg_const_val32(instr, Op1_num);
			if(op0->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op0->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op0->type != Instr::REG && op0->type != Instr::MREF) {
			int Op1_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op1_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op1->type == Instr::REG || op1->type == Instr::MREF) {
			int Op2_num = (int)op1->value[0];
			nvbit_add_call_arg_const_val32(instr, Op2_num);
			if(op1->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op1->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op1->type != Instr::REG && op1->type != Instr::MREF) {
			int Op2_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op2_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op2->type == Instr::REG || op2->type == Instr::MREF) {
			int Op3_num = (int)op2->value[0];
			nvbit_add_call_arg_const_val32(instr, Op3_num);
			if(op2->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op2->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op2->type != Instr::REG && op2->type != Instr::MREF) {
			int Op3_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op3_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op3->type == Instr::REG || op3->type == Instr::MREF) {
			int Op4_num = (int)op3->value[0];
			nvbit_add_call_arg_const_val32(instr, Op4_num);
			if(op3->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op3->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op3->type != Instr::REG && op3->type != Instr::MREF) {
			int Op4_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op4_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	}


	if(opera_number == 5) {
		const Instr::operand_t *op0 = instr->getOperand(0);
		const Instr::operand_t *op1 = instr->getOperand(1);
		const Instr::operand_t *op2 = instr->getOperand(2);
		const Instr::operand_t *op3 = instr->getOperand(3);
		const Instr::operand_t *op4 = instr->getOperand(4);
		nvbit_insert_call(instr, "five_op_instr_process", IPOINT_BEFORE);
		nvbit_add_call_arg_pred_val(instr);
		if (op0->type == Instr::REG || op0->type == Instr::MREF) {
			int Op1_num = (int)op0->value[0];
			nvbit_add_call_arg_const_val32(instr, Op1_num);
			if(op0->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op0->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op0->type != Instr::REG && op0->type != Instr::MREF) {
			int Op1_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op1_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op1->type == Instr::REG || op1->type == Instr::MREF) {
			int Op2_num = (int)op1->value[0];
			nvbit_add_call_arg_const_val32(instr, Op2_num);
			if(op1->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op1->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op1->type != Instr::REG && op1->type != Instr::MREF) {
			int Op2_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op2_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op2->type == Instr::REG || op2->type == Instr::MREF) {
			int Op3_num = (int)op2->value[0];
			nvbit_add_call_arg_const_val32(instr, Op3_num);
			if(op2->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op2->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op2->type != Instr::REG && op2->type != Instr::MREF) {
			int Op3_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op3_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op3->type == Instr::REG || op3->type == Instr::MREF) {
			int Op4_num = (int)op3->value[0];
			nvbit_add_call_arg_const_val32(instr, Op4_num);
			if(op3->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op3->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op3->type != Instr::REG && op3->type != Instr::MREF) {
			int Op4_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op4_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op4->type == Instr::REG || op4->type == Instr::MREF) {
			int Op5_num = (int)op4->value[0];
			nvbit_add_call_arg_const_val32(instr, Op5_num);
			if(op4->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op4->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op4->type != Instr::REG && op4->type != Instr::MREF) {
			int Op5_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op5_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	}
	if(opera_number == 6) {
		const Instr::operand_t *op0 = instr->getOperand(0);
		const Instr::operand_t *op1 = instr->getOperand(1);
		const Instr::operand_t *op2 = instr->getOperand(2);
		const Instr::operand_t *op3 = instr->getOperand(3);
		const Instr::operand_t *op4 = instr->getOperand(4);
		const Instr::operand_t *op5 = instr->getOperand(5);
		nvbit_insert_call(instr, "six_op_instr_process", IPOINT_BEFORE);
		nvbit_add_call_arg_pred_val(instr);
		if (op0->type == Instr::REG || op0->type == Instr::MREF) {
			int Op1_num = (int)op0->value[0];
			nvbit_add_call_arg_const_val32(instr, Op1_num);
			if(op0->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op0->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op0->type != Instr::REG && op0->type != Instr::MREF) {
			int Op1_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op1_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op1->type == Instr::REG || op1->type == Instr::MREF) {
			int Op2_num = (int)op1->value[0];
			nvbit_add_call_arg_const_val32(instr, Op2_num);
			if(op1->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op1->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op1->type != Instr::REG && op1->type != Instr::MREF) {
			int Op2_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op2_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op2->type == Instr::REG || op2->type == Instr::MREF) {
			int Op3_num = (int)op2->value[0];
			nvbit_add_call_arg_const_val32(instr, Op3_num);
			if(op2->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op2->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op2->type != Instr::REG && op2->type != Instr::MREF) {
			int Op3_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op3_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op3->type == Instr::REG || op3->type == Instr::MREF) {
			int Op4_num = (int)op3->value[0];
			nvbit_add_call_arg_const_val32(instr, Op4_num);
			if(op3->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op3->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op3->type != Instr::REG && op3->type != Instr::MREF) {
			int Op4_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op4_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op4->type == Instr::REG || op4->type == Instr::MREF) {
			int Op5_num = (int)op4->value[0];
			nvbit_add_call_arg_const_val32(instr, Op5_num);
			if(op4->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op4->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op4->type != Instr::REG && op4->type != Instr::MREF) {
			int Op5_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op5_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	 	if (op5->type == Instr::REG || op5->type == Instr::MREF) {
			int Op6_num = (int)op5->value[0];
			nvbit_add_call_arg_const_val32(instr, Op6_num);
			if(op5->type == Instr::REG){
				int type = 1;
				nvbit_add_call_arg_const_val32(instr, type);
			}
			if(op5->type == Instr::MREF){ 
				int type = 2;
				nvbit_add_call_arg_const_val32(instr, type);
			}
		}	
		if (op5->type != Instr::REG && op5->type != Instr::MREF) {
			int Op6_num = 255;
			nvbit_add_call_arg_const_val32(instr, Op6_num);
			int type = 3;
			nvbit_add_call_arg_const_val32(instr, type); 
		}
	}
    }
    gettimeofday(&timeEnd, NULL); 
    runTime = (timeEnd.tv_sec - timeStart.tv_sec ) + (double)(timeEnd.tv_usec - timeStart.tv_usec)/1000000;  
    printf("[1m[40;36mRunTime is %lf seconds.[0m[0m\n", runTime);
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    
    /* Identify all the possible CUDA launch events */
    
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel) {
        cuLaunch_params *p = (cuLaunch_params *)params;
	    
	if (is_exit) {
		pthread_mutex_lock(&mutex);
		if (kernel_id >= ker_begin_interval && kernel_id < ker_end_interval) {
			nvbit_enable_instrumented(ctx, p->f, true);
		}else{
		nvbit_enable_instrumented(ctx, p->f, false);
		}
		evecounter = 0;
        }else{
		CUDA_SAFECALL(cudaDeviceSynchronize());
		tot_app_instrs += evecounter;
		int num_ctas = 0;
		if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {
			cuLaunchKernel_params *p2 = (cuLaunchKernel_params *)params;
			num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
		}
		printf( "kernelid = %d , thread-blocks %d\n", kernel_id, num_ctas);
		
		pthread_mutex_unlock(&mutex);
	}
    }
}
