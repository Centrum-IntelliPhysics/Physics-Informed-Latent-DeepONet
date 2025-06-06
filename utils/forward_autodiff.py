import functorch

#===================================================#
# Using forward automatic differention to estimate derivatives in the physics informed loss

# Function to compute first-order derivative using jvp
def FWDAD_first_order_derivative(f, primals, tangents):
    _, tangents_out = functorch.jvp(f, (primals,), (tangents,))
    return tangents_out

# Function to compute second-order derivative using jvp
def FWDAD_second_order_derivative(f, primals, tangents):
    g = lambda primals: functorch.jvp(f, (primals,), (tangents,))[1]
    _, tangents_out = functorch.jvp(g, (primals,), (tangents,))
    return tangents_out
#===================================================#