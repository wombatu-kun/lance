// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use crate::dataset::transaction::{Operation, Transaction, UpdateMap, UpdateMapEntry};

use super::Dataset;
use crate::Result;
use futures::future::BoxFuture;
use lance_core::datatypes::FieldRef;
use lance_core::datatypes::Schema;

/// Execute a metadata update operation on a dataset.
/// This is moved from Dataset::update_op to keep metadata logic in this module.
pub async fn execute_metadata_update(dataset: &mut Dataset, operation: Operation) -> Result<()> {
    let transaction = Transaction::new(dataset.manifest.version, operation, None);
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;
    Ok(())
}

/// Builder for metadata update operations that supports optional replace semantics.
/// This provides backward compatibility while adding new functionality.
pub struct UpdateMetadataBuilder<'a> {
    dataset: &'a mut Dataset,
    values: Vec<UpdateMapEntry>,
    replace: bool,
    metadata_type: MetadataType,
}

/// Type of metadata being updated
pub enum MetadataType {
    Config,
    TableMetadata,
    SchemaMetadata,
}

impl<'a> UpdateMetadataBuilder<'a> {
    pub fn new(
        dataset: &'a mut Dataset,
        values: impl IntoIterator<Item = impl Into<UpdateMapEntry>>,
        metadata_type: MetadataType,
    ) -> Self {
        Self {
            dataset,
            values: values.into_iter().map(Into::into).collect(),
            replace: false,
            metadata_type,
        }
    }

    /// Set the replace flag to true, causing the entire metadata map to be replaced
    /// instead of merged.
    pub fn replace(mut self) -> Self {
        self.replace = true;
        self
    }

    fn create_update_map(values: Vec<UpdateMapEntry>, replace: bool) -> UpdateMap {
        UpdateMap {
            update_entries: values,
            replace,
        }
    }
}

impl<'a> std::future::IntoFuture for UpdateMetadataBuilder<'a> {
    type Output = Result<HashMap<String, String>>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let update_map = Self::create_update_map(self.values, self.replace);

            let operation = match self.metadata_type {
                MetadataType::Config => Operation::UpdateConfig {
                    config_updates: Some(update_map),
                    table_metadata_updates: None,
                    schema_metadata_updates: None,
                    field_metadata_updates: HashMap::new(),
                },
                MetadataType::TableMetadata => Operation::UpdateConfig {
                    config_updates: None,
                    table_metadata_updates: Some(update_map),
                    schema_metadata_updates: None,
                    field_metadata_updates: HashMap::new(),
                },
                MetadataType::SchemaMetadata => Operation::UpdateConfig {
                    config_updates: None,
                    table_metadata_updates: None,
                    schema_metadata_updates: Some(update_map),
                    field_metadata_updates: HashMap::new(),
                },
            };

            execute_metadata_update(self.dataset, operation).await?;

            // Get result after the update
            let result = match self.metadata_type {
                MetadataType::Config => self.dataset.manifest.config.clone(),
                MetadataType::TableMetadata => self.dataset.manifest.table_metadata.clone(),
                MetadataType::SchemaMetadata => self.dataset.manifest.schema.metadata.clone(),
            };

            Ok(result)
        })
    }
}

#[derive(Debug)]
pub struct UpdateFieldMetadataBuilder<'a> {
    dataset: &'a mut Dataset,
    field_metadata_updates: HashMap<i32, UpdateMap>,
}

impl<'a> UpdateFieldMetadataBuilder<'a> {
    pub fn new(dataset: &'a mut Dataset) -> Self {
        Self {
            dataset,
            field_metadata_updates: HashMap::new(),
        }
    }

    fn apply<'b>(
        mut self,
        field: impl Into<FieldRef<'b>> + 'b,
        values: impl IntoIterator<Item = impl Into<UpdateMapEntry>>,
        replace: bool,
    ) -> Result<Self> {
        let field_id = field.into().into_id(self.dataset.schema())?;
        let values = UpdateMap {
            update_entries: values.into_iter().map(Into::into).collect(),
            replace,
        };
        self.field_metadata_updates.insert(field_id, values);
        Ok(self)
    }
    pub fn update<'b>(
        self,
        field: impl Into<FieldRef<'b>> + 'b,
        values: impl IntoIterator<Item = impl Into<UpdateMapEntry>>,
    ) -> Result<Self> {
        self.apply(field, values, false)
    }

    pub fn replace<'b>(
        self,
        field: impl Into<FieldRef<'b>> + 'b,
        values: impl IntoIterator<Item = impl Into<UpdateMapEntry>>,
    ) -> Result<Self> {
        self.apply(field, values, true)
    }
}

impl<'a> std::future::IntoFuture for UpdateFieldMetadataBuilder<'a> {
    type Output = Result<&'a Schema>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            execute_metadata_update(
                self.dataset,
                Operation::UpdateConfig {
                    config_updates: None,
                    table_metadata_updates: None,
                    schema_metadata_updates: None,
                    field_metadata_updates: self.field_metadata_updates,
                },
            )
            .await?;
            Ok(self.dataset.schema())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lance_core::Error;
    use lance_datagen::{BatchCount, RowCount, array, gen_batch};
    use rstest::rstest;

    use super::*;
    use arrow_array::{
        ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, UInt32Array, types::Int32Type,
    };
    use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};

    #[rstest]
    #[tokio::test]
    async fn test_update_config() {
        let data = gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();

        // Insert
        let mut desired_config = dataset.manifest.config.clone();
        desired_config.insert("lance.test".to_string(), "value".to_string());
        desired_config.insert("other-key".to_string(), "other-value".to_string());

        dataset
            .update_config([("lance.test", "value"), ("other-key", "other-value")])
            .await
            .unwrap();
        assert_eq!(dataset.manifest.config, desired_config);
        assert_eq!(dataset.config(), &desired_config);

        // Update and delete
        let mut desired_config = dataset.manifest.config.clone();
        desired_config.insert("other-key".to_string(), "new-value".to_string());
        desired_config.remove("lance.test");

        dataset
            .update_config([("other-key", Some("new-value")), ("lance.test", None)])
            .await
            .unwrap();

        // Replace
        let desired_config = HashMap::from_iter([
            ("k1".to_string(), "v1".to_string()),
            ("k2".to_string(), "v2".to_string()),
        ]);
        dataset
            .update_config([("k1", "v1"), ("k2", "v2")])
            .replace()
            .await
            .unwrap();
        assert_eq!(dataset.config(), &desired_config);

        // Clear
        dataset
            .update_config([] as [UpdateMapEntry; 0])
            .replace()
            .await
            .unwrap();
        assert!(dataset.config().is_empty());
    }

    #[tokio::test]
    async fn test_update_table_metadata() {
        let data = gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();

        // Insert
        let mut desired_table_meta = dataset.manifest.table_metadata.clone();
        desired_table_meta.insert("lance.table".to_string(), "value".to_string());
        desired_table_meta.insert(
            "other-table-key".to_string(),
            "other-table-value".to_string(),
        );

        dataset
            .update_metadata([
                ("lance.table", "value"),
                ("other-table-key", "other-table-value"),
            ])
            .await
            .unwrap();
        assert_eq!(dataset.manifest.table_metadata, desired_table_meta);

        // Update and delete
        let mut desired_table_meta = dataset.manifest.table_metadata.clone();
        desired_table_meta.insert("other-table-key".to_string(), "new-table-value".to_string());
        desired_table_meta.remove("lance.table");

        dataset
            .update_metadata([
                ("other-table-key", Some("new-table-value")),
                ("lance.table", None),
            ])
            .await
            .unwrap();

        // Replace
        let desired_table_meta = HashMap::from_iter([
            ("k1".to_string(), "v1".to_string()),
            ("k2".to_string(), "v2".to_string()),
        ]);
        dataset
            .update_metadata([("k1", "v1"), ("k2", "v2")])
            .replace()
            .await
            .unwrap();
        assert_eq!(dataset.manifest.table_metadata, desired_table_meta);

        // Clear
        dataset
            .update_metadata([] as [UpdateMapEntry; 0])
            .replace()
            .await
            .unwrap();
        assert!(dataset.manifest.table_metadata.is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn test_replace_schema_metadata_preserves_fragments() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::UInt32,
            false,
        )]));

        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(0..100))],
        );

        let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        let manifest_before = dataset.manifest.clone();

        let mut new_schema_meta = HashMap::new();
        new_schema_meta.insert("new_key".to_string(), "new_value".to_string());
        #[allow(deprecated)]
        dataset
            .replace_schema_metadata(new_schema_meta.clone())
            .await
            .unwrap();

        let manifest_after = dataset.manifest.clone();

        assert_eq!(manifest_before.fragments, manifest_after.fragments);
    }

    #[rstest]
    #[tokio::test]
    async fn test_replace_fragment_metadata_preserves_fragments() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::UInt32,
            false,
        )]));

        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(UInt32Array::from_iter_values(0..100))],
        );

        let reader = RecordBatchIterator::new(vec![data.unwrap()].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        let manifest_before = dataset.manifest.clone();

        let mut new_field_meta = HashMap::new();
        new_field_meta.insert("new_key".to_string(), "new_value".to_string());
        dataset
            .replace_field_metadata(vec![(0, new_field_meta.clone())])
            .await
            .unwrap();

        let manifest_after = dataset.manifest.clone();

        assert_eq!(manifest_before.fragments, manifest_after.fragments);
    }

    async fn test_dataset_nested() -> Dataset {
        let schema = Arc::new(ArrowSchema::new_with_metadata(
            vec![
                ArrowField::new("id", DataType::Int32, false),
                ArrowField::new("name", DataType::Utf8, true),
                ArrowField::new(
                    "nested",
                    DataType::Struct(Fields::from(vec![
                        ArrowField::new("sub_field", DataType::Int32, true),
                        ArrowField::new("another_field", DataType::Float32, false),
                    ])),
                    true,
                ),
            ],
            Default::default(),
        ));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(arrow_array::StringArray::from(vec!["a", "b", "c"])),
                Arc::new(arrow_array::StructArray::from(vec![
                    (
                        Arc::new(ArrowField::new("sub_field", DataType::Int32, true)),
                        Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef,
                    ),
                    (
                        Arc::new(ArrowField::new("another_field", DataType::Float32, false)),
                        Arc::new(arrow_array::Float32Array::from(vec![1.0, 2.0, 3.0])) as ArrayRef,
                    ),
                ])),
            ],
        )
        .unwrap();

        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
            "memory://test",
            None,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_update_field_metadata_by_path() {
        let mut dataset = test_dataset_nested().await;

        // Test updating metadata by field path
        dataset
            .update_field_metadata()
            .update("name", [("key1", "value1"), ("key2", "value2")])
            .unwrap()
            .await
            .unwrap();

        // Verify metadata was updated
        let field = dataset.schema().field("name").unwrap();
        assert_eq!(field.metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(field.metadata.get("key2"), Some(&"value2".to_string()));

        // Test updating nested field by path
        dataset
            .update_field_metadata()
            .update("nested.sub_field", [("nested_key", "nested_value")])
            .unwrap()
            .await
            .unwrap();

        let nested_field = dataset.schema().field("nested.sub_field").unwrap();
        assert_eq!(
            nested_field.metadata.get("nested_key"),
            Some(&"nested_value".to_string())
        );
    }

    #[tokio::test]
    async fn test_update_field_metadata_by_id() {
        let mut dataset = test_dataset_nested().await;

        // Get field IDs first
        let id_field_id = dataset.schema().field("id").unwrap().id;
        let name_field_id = dataset.schema().field("name").unwrap().id;

        // Test updating metadata by field ID
        dataset
            .update_field_metadata()
            .update(id_field_id, [("id_key", "id_value")])
            .unwrap()
            .await
            .unwrap();

        let field = dataset.schema().field_by_id(id_field_id).unwrap();
        assert_eq!(field.metadata.get("id_key"), Some(&"id_value".to_string()));

        // Update another field by ID
        dataset
            .update_field_metadata()
            .update(name_field_id, [("val_key", "val_value")])
            .unwrap()
            .await
            .unwrap();

        let field = dataset.schema().field_by_id(name_field_id).unwrap();
        assert_eq!(
            field.metadata.get("val_key"),
            Some(&"val_value".to_string())
        );
    }

    #[tokio::test]
    async fn test_update_field_metadata_replace() {
        let mut dataset = test_dataset_nested().await;

        // First, add some metadata using update
        dataset
            .update_field_metadata()
            .update("id", [("key1", "value1"), ("key2", "value2")])
            .unwrap()
            .await
            .unwrap();

        let field = dataset.schema().field("id").unwrap();
        assert_eq!(field.metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(field.metadata.get("key2"), Some(&"value2".to_string()));

        // Now replace the metadata
        dataset
            .update_field_metadata()
            .replace("id", [("new_key", "new_value")])
            .unwrap()
            .await
            .unwrap();

        let field = dataset.schema().field("id").unwrap();
        // Old keys should be gone
        assert_eq!(field.metadata.get("key1"), None);
        assert_eq!(field.metadata.get("key2"), None);
        // New key should be present
        assert_eq!(
            field.metadata.get("new_key"),
            Some(&"new_value".to_string())
        );

        // Test clearing metadata completely by replacing with empty array
        dataset
            .update_field_metadata()
            .replace("id", [] as [(&str, &str); 0])
            .unwrap()
            .await
            .unwrap();

        let field = dataset.schema().field("id").unwrap();
        assert!(field.metadata.is_empty());
    }

    #[tokio::test]
    async fn test_update_field_metadata_invalid_path() {
        let mut dataset = test_dataset_nested().await;

        // Test updating non-existent field by path
        let result = dataset
            .update_field_metadata()
            .update("non_existent_field", [("key", "value")]);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::FieldNotFound { .. }));
        assert!(
            err.to_string()
                .contains("Field 'non_existent_field' not found"),
            "Expected error message to contain field name, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_update_field_metadata_invalid_id() {
        let mut dataset = test_dataset_nested().await;

        // Test updating with invalid field ID
        // Use an ID that's definitely invalid
        let invalid_id = 99999;

        // Create a builder and try to execute - this should eventually fail somewhere
        let result = async {
            dataset
                .update_field_metadata()
                .update(invalid_id, [("key", "value")])?
                .await
        }
        .await;

        assert!(matches!(result, Err(Error::InvalidInput { .. })));
    }

    /// Helper to create a simple dataset with a non-nullable `id` field suitable for PK tests.
    async fn test_dataset_for_pk() -> Dataset {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("value", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(arrow_array::StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
            "memory://",
            None,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_update_field_metadata_sets_unenforced_primary_key() {
        let mut dataset = test_dataset_for_pk().await;

        // Legacy boolean flag should map to position 0.
        dataset
            .update_field_metadata()
            .update("id", [("lance-schema:unenforced-primary-key", "true")])
            .unwrap()
            .await
            .unwrap();

        let field = dataset.schema().field("id").unwrap();
        assert!(
            field.is_unenforced_primary_key(),
            "Field should be recognized as unenforced primary key after metadata update"
        );
        assert_eq!(
            field.unenforced_primary_key_position,
            Some(0),
            "Legacy boolean flag should map to position 0"
        );

        // Explicit position should override the legacy flag.
        dataset
            .update_field_metadata()
            .update(
                "id",
                [("lance-schema:unenforced-primary-key:position", "2")],
            )
            .unwrap()
            .await
            .unwrap();

        let field = dataset.schema().field("id").unwrap();
        assert!(field.is_unenforced_primary_key());
        assert_eq!(
            field.unenforced_primary_key_position,
            Some(2),
            "Explicit position should take precedence over the legacy boolean flag"
        );
    }

    #[tokio::test]
    async fn test_update_field_metadata_primary_key_used_by_merge_insert() {
        use crate::dataset::write::merge_insert::*;

        let mut dataset = test_dataset_for_pk().await;

        // Set PK via metadata update (the bug scenario)
        dataset
            .update_field_metadata()
            .update("id", [("lance-schema:unenforced-primary-key", "true")])
            .unwrap()
            .await
            .unwrap();

        let dataset = Arc::new(dataset);

        // Prepare new data that overlaps with existing
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("value", DataType::Utf8, true),
        ]));
        let new_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2, 4])),
                Arc::new(arrow_array::StringArray::from(vec!["updated", "new"])),
            ],
        )
        .unwrap();

        // MergeInsert with empty `on` keys — should default to the unenforced PK
        let mut builder = MergeInsertBuilder::try_new(dataset.clone(), Vec::new()).unwrap();
        builder
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll);
        let job = builder.try_build().unwrap();

        let new_reader = Box::new(RecordBatchIterator::new([Ok(new_batch)], schema.clone()));
        let new_stream = lance_datafusion::utils::reader_to_stream(new_reader);

        let (updated_dataset, stats) = job.execute(new_stream).await.unwrap();

        assert_eq!(stats.num_inserted_rows, 1, "id=4 should be inserted");
        assert_eq!(stats.num_updated_rows, 1, "id=2 should be updated");

        let result = updated_dataset.scan().try_into_batch().await.unwrap();
        let ids = result
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let values = result
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();

        let mut pairs: Vec<(i32, String)> = (0..ids.len())
            .map(|i| (ids.value(i), values.value(i).to_string()))
            .collect();
        pairs.sort_by_key(|(id, _)| *id);

        assert_eq!(
            pairs,
            vec![
                (1, "a".to_string()),
                (2, "updated".to_string()),
                (3, "c".to_string()),
                (4, "new".to_string()),
            ]
        );
    }
}
