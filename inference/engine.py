import time
from typing import Dict, List
import acl
import numpy as np

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_NORMAL_ONLY = 2
NPY_FLOAT32 = 11
def check_ret(str,ret):
    if ret != 0:
        print(f"return code is {ret}, detail: {str}",flush=True) 

def initResource(device):
    ret = acl.init()
    check_ret("init", ret)
    ret = acl.rt.set_device(device)
    check_ret("set_device", ret)
    context,ret = acl.rt.create_context(device)
    check_ret("create_context", ret)
    return context

def destroyResource(device,context):
    ret = acl.rt.reset_device(device)
    ret = acl.finalize()
    ret = acl.rt.destroy_context(context)

dtype2NpType = {0:np.float32,1:np.float16,2:np.int8,3:np.int32,9:np.int64}

class ACLModel:
    def __init__(self,model_path,context=None,callback=None):
        self.context = context
        self.model_id = None
        self.model_desc = None
        self.callback_func = callback 
        self.tid = None
        self.stream = None
        self.callback_interval = 1
        self.exit_flag = False
        self.input_dataset, self.output_dataset = None, None
        self.inputs:List[Dict[str,]] = []
        self.outputs:List[Dict[str,]] =  []
        self.loadModel(model_path)
        self.allocateMem()
        if not callback:
            return
        self.stream, ret = acl.rt.create_stream()
        self.tid, ret = acl.util.start_thread(self._process_callback,
                                         [self.context, 50])
        check_ret("acl.util.start_thread", ret)
        ret = acl.rt.subscribe_report(self.tid, self.stream)
        check_ret("acl.rt.subscribe_report", ret)
    
    def unload(self):
        if self.callback_func:
            ret = acl.rt.synchronize_stream(self.stream)
            # 2.7 取消线程注册，Stream上的回调函数不再由指定线程处理。
            ret = acl.rt.unsubscribe_report(self.tid, self.stream)
            self.exit_flag = True
            ret = acl.util.stop_thread(self.tid)
            ret = acl.rt.destroy_stream(self.stream)
        self.freeMem()
        self.unloadModel()


    def loadModel(self, model_path):
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        check_ret("load model",ret)
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("get model desc",ret)
    
    def unloadModel(self):
        ret = acl.mdl.unload(self.model_id)
        if self.model_desc:
            ret  = acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

    def allocateMem(self):
        self.input_dataset = acl.mdl.create_dataset()
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        self.inputs = []
        for i in range(input_size):
            buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("alloc input memory",ret)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, data)
            check_ret("add_dataset_buffer",ret)
            self.inputs.append({"buffer": buffer, "size": buffer_size})

        self.output_dataset = acl.mdl.create_dataset()
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self.outputs = []
        for i in range(output_size):
            buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            data_type = acl.mdl.get_output_data_type(self.model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("alloc output memory",ret)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, data)
            check_ret("add_dataset_buffer",ret)
            buffer_host, ret = acl.rt.malloc_host(buffer_size)
            check_ret("alloc output host memory",ret)
            self.outputs.append({"buffer": buffer, "size": buffer_size,'buffer_host':buffer_host,'dtype':dtype2NpType[data_type]})

    def freeMem(self):
        for item in self.input_data:
            ret = acl.rt.free(item["buffer"])
        ret = acl.mdl.destroy_dataset(self.input_dataset)
        for item in self.output_data:
            ret = acl.rt.free(item["buffer"])
            ret = acl.rt.free_host(item["buffer_host"])
        ret = acl.mdl.destroy_dataset(self.output_dataset)

    def inference(self,datas) -> List[np.ndarray]:
        for i,data in enumerate(datas):
            bytes_data = data.tobytes()
            np_ptr = acl.util.bytes_to_ptr(bytes_data)
            ret = acl.rt.memcpy(self.inputs[i]["buffer"], self.inputs[i]["size"], np_ptr,self.inputs[i]["size"], ACL_MEMCPY_HOST_TO_DEVICE)
            check_ret("memcpy", ret)
        ret = acl.mdl.execute(self.model_id, self.input_dataset,self.output_dataset)  
        inference_result = []
        for idx,out in enumerate(self.outputs):
            ret = acl.rt.memcpy(out['buffer_host'], out["size"],out["buffer"],out["size"],ACL_MEMCPY_DEVICE_TO_HOST)
            bytes_out = acl.util.ptr_to_bytes(out['buffer_host'], out["size"])
            dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, idx)
            out_data = np.frombuffer(bytes_out, dtype=out['dtype']).reshape(dims['dims'])
            inference_result.append(out_data)
        return inference_result
    
    def inference_async(self,datas,other_args) -> List[np.ndarray]:
        for i,data in enumerate(datas):
            np_ptr = acl.util.bytes_to_ptr(data.tobytes())
            ret = acl.rt.memcpy(self.inputs[i]["buffer"], self.inputs[i]["size"], np_ptr,self.inputs[i]["size"], ACL_MEMCPY_HOST_TO_DEVICE)
            check_ret("memcpy", ret)
        ret = acl.mdl.execute_async(self.model_id, self.input_dataset,self.output_dataset,self.stream)
        check_ret("exec_async", ret)
        print(f"submit exec task {other_args[1]}")
        ret = acl.rt.launch_callback(self.callPostProcess,other_args,1,self.stream)
        check_ret("launch callback", ret)

    def _process_callback(self, args_list):
        context, timeout = args_list
        acl.rt.set_context(context)
        while self.callback_interval:
            acl.rt.process_report(timeout)
            if self.exit_flag:
                print("[Callback] exit acl.rt.process_report")
                break

    def callPostProcess(self,other_args):
        print("start callback",flush=True)
        time1 = time.time()
        inference_result = []
        for idx,out in enumerate(self.outputs):
            ret = acl.rt.memcpy(out['buffer_host'], out["size"],out["buffer"],out["size"],ACL_MEMCPY_DEVICE_TO_HOST)
            bytes_out = acl.util.ptr_to_bytes(out['buffer_host'], out["size"])
            dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, idx)
            data = np.frombuffer(bytes_out, dtype=out['dtype']).reshape(dims['dims'])
            inference_result.append(data)
        if not self.callback_func:
            return
        self.callback_func(inference_result,other_args)
        print(f"end callback, use time: {time.time()-time1}")
