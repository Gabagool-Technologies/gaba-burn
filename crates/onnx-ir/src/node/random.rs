use crate::ir::{ArgType, ElementType, Node, TensorType};
use crate::protos::tensor_proto::DataType;

/// Update output rank for Random operations with explicit shape attribute.
pub fn random_update_output(node: &mut Node) {
    log::debug!("Random rank inference for node {}", node.name);

    let dtype = node
        .attrs
        .get("dtype")
    .map(|val| DataType::from(val.clone().into_i32()))
    .unwrap_or(DataType::Float);
    log::debug!("Random dtype for {}: {:?}", node.name, dtype);

    let shape = node
        .attrs
        .get("shape")
        .expect("required shape attribute missing")
        .clone()
        .into_i64s();
    log::debug!("Random shape for {}: {:?}", node.name, shape);

    let elem_type = match dtype {
        DataType::Float => ElementType::Float32,
        DataType::Double => ElementType::Float64,
        _ => panic!("tensor with type {dtype:?} not supported for random output"),
    };

    let rank = shape.len();
    log::debug!("Random output rank for {}: {}", node.name, rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type,
        rank,
        static_shape: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;
    use crate::protos::tensor_proto::DataType;

    fn create_test_node(dtype: i32, shape: Vec<i64>) -> Node {
        NodeBuilder::new(NodeType::RandomNormal, "test_random")
            .output_tensor_f32("output", 0, None) // Rank 0 will be updated
            .attr_int("dtype", dtype as i64)
            .attr_ints("shape", shape)
            .build()
    }

    #[test]
    fn test_random_normal_float() {
    let mut node = create_test_node(i32::from(DataType::Float), vec![2, 3, 4]);
        random_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_random_normal_double() {
    let mut node = create_test_node(i32::from(DataType::Double), vec![5]);
        random_update_output(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float64);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    #[should_panic(expected = "required shape attribute missing")]
    fn test_random_normal_missing_shape() {
        // Create node and then manually remove the shape attribute
    let mut node = create_test_node(i32::from(DataType::Float), vec![2, 3]);
        node.attrs.remove("shape");
        random_update_output(&mut node);
    }

    #[test]
    #[should_panic(expected = "tensor with type DataType::Int32 not supported for random output")]
    fn test_random_normal_unsupported_type() {
    let mut node = create_test_node(i32::from(DataType::Int32), vec![2, 3]);
        random_update_output(&mut node);
    }
}
