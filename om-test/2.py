import acl
def init_resource(device_id):
    ret = acl.init()
    ret = acl.rt.set_device(device_id)
    context,ret = acl.rt.create_context(device_id)
    return context

context = init_resource(0)
print(acl.mdl.query_size("/root/program/mlp.om"))