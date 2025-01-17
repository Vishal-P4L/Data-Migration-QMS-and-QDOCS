import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np

# Title for the Streamlit app
st.title('CSV/Excel File Upload and Multi-Value Data Filtering')

# File uploader widget for the main file
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"], key="main_file")

# If a file is uploaded
if uploaded_file is not None:
    # Check file type and load as appropriate
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview (without index):")
        st.dataframe(df.style.hide(axis="index"))
        # Set the download file name based on the uploaded file name
        download_file_name = uploaded_file.name.split('.')[0] + "_output.csv"
    else:
        # Load Excel file
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Select a sheet", excel_file.sheet_names)
        df = excel_file.parse(sheet_name)
        st.write("Data Preview (without index):")
        st.dataframe(df.style.hide(axis="index"))
        # Set the download file name based on the selected sheet name
        download_file_name = sheet_name + "_output.csv"

    # Check if "Transformation Logic" exists and filter rows based on its value
    if "Transformation Logic" in df.columns:
        transformation_logic_filtered = df[df["Transformation Logic"].str.contains(
            "If value is not present map to Owning Department", na=False)]
        print('transformation_logic_filtered', transformation_logic_filtered)

        if not transformation_logic_filtered.empty:
            st.write("Rows with 'Transformation Logic' containing 'If value is not present map to Owning Department':")
            st.dataframe(
                transformation_logic_filtered[["Source Field Name", "Transformation Logic"]].style.hide(axis="index"))
        else:
            st.warning("No rows found with the specified 'Transformation Logic' value.")
    else:
        st.warning("'Transformation Logic' column is not present in the uploaded file.")

    # Create a dropdown for column selection
    column_name = st.selectbox('Select a column to filter by:', df.columns)

    # Create a multiselect for selecting multiple values based on the selected column
    column_values = df[column_name].unique()
    selected_values = st.multiselect(f'Select values from the column {column_name}:', column_values)

    # Apply the filter to the DataFrame if any values are selected
    if selected_values:
        filtered_df = df[df[column_name].isin(selected_values)].reset_index(drop=True)

        # Select only the desired columns after filtering
        filtered_df = filtered_df[['Source Field Name', 'Target Field Name', 'Migrate?', 'Transformation Logic']]

        st.write("Data Preview After Filtering:")
        st.dataframe(filtered_df.style.hide(axis="index"))

        # Save all target fields and filtered target fields
        all_target_list = df['Target Field Name'].unique()
        filtered_target_list = filtered_df['Target Field Name'].unique()

        # Compare the lists and handle missing columns
        missing_columns = [col for col in all_target_list if col not in filtered_target_list]
        for col in missing_columns:
            filtered_df[col] = "N/A"

        # Categorize into matched, unmatched, and missing source
        matchedColumns = filtered_df[filtered_df['Source Field Name'] == filtered_df['Target Field Name']]
        unmatchedColumns = filtered_df[(filtered_df['Source Field Name'] != filtered_df['Target Field Name']) &
                                       (~filtered_df['Source Field Name'].isna()) &
                                       (~filtered_df['Target Field Name'].isna())]
        missingSourceColumns = filtered_df[
            filtered_df['Source Field Name'].isna() & filtered_df['Target Field Name'].notna()]

        st.write("Matched Columns:")
        st.dataframe(matchedColumns.style.hide(axis="index"))
        st.write("Unmatched Columns:")
        st.dataframe(unmatchedColumns.style.hide(axis="index"))
        st.write("Missing Source Columns:")
        st.dataframe(missingSourceColumns.style.hide(axis="index"))

# File upload for Source and Target files
st.subheader("Upload Source and Target Files")

source_file = st.file_uploader("Choose a Source CSV file", type="csv", key="source_file")
target_file = st.file_uploader("Choose a Target CSV file", type="csv", key="target_file")

if source_file and target_file:
    source_df = pd.read_csv(source_file)
    target_df = pd.read_csv(target_file)
    result_df = pd.DataFrame()

    # Handle unmatched columns
    if not unmatchedColumns.empty:
        unmatched_mappings = unmatchedColumns[['Source Field Name', 'Target Field Name', 'Transformation Logic']]
        for _, row in unmatched_mappings.iterrows():
            source_col = row['Source Field Name']
            target_col = row['Target Field Name']
            transformation_logic = row['Transformation Logic']

            # Check if target column is 'training_impact__c'
            if target_col == 'training_impact__c':
                if pd.notna(transformation_logic):
                    # Apply transformation logic for 'training_impact__c'
                    result_df[target_col] = [transformation_logic] * len(source_df)
                    st.write(f"Filled '{target_col}' with transformation logic: {transformation_logic}.")
                else:
                    # If no transformation logic, fill with the source column values
                    if source_col in source_df.columns:
                        result_df[target_col] = source_df[source_col]
                        st.write(
                            f"Values from '{source_col}' in source file have been added to '{target_col}' in the target file.")
                    else:
                        # If source column is missing, fill with empty or default value
                        result_df[target_col] = [""] * len(source_df)
                        st.warning(
                            f"Source column '{source_col}' is missing in the source data. Filled '{target_col}' with empty values.")

            else:
                # Handle other columns (general case)
                if source_col in source_df.columns:
                    # If source column exists in source_df, assign values to the target column
                    result_df[target_col] = source_df[source_col]
                    st.write(
                        f"Values from '{source_col}' in source file have been added to '{target_col}' in the target file.")
                else:
                    # If source column is missing, apply the transformation logic
                    st.warning(
                        f"Source column '{source_col}' is missing in the source data. Filling '{target_col}' with transformation logic.")
                    result_df[target_col] = [transformation_logic if pd.notna(transformation_logic) else ""] * len(
                        source_df)

    # Handle matched columns
    if not matchedColumns.empty:
        for _, row in matchedColumns.iterrows():
            source_col = row['Source Field Name']
            target_col = row['Target Field Name']
            transformation_logic = row['Transformation Logic']

            if source_col in source_df.columns:
                # Apply transformation logic or use source data as is
                result_df[target_col] = source_df[source_col].apply(
                    lambda x: transformation_logic if pd.notna(transformation_logic) else x
                )
            else:
                # If the source column is missing, fill with transformation logic or default value
                st.warning(
                    f"Source column {source_col} is missing in the source data. Filling with transformation logic.")
                result_df[target_col] = [transformation_logic if pd.notna(transformation_logic) else ""] * len(
                    source_df)

    # Process missing source columns
    if not missingSourceColumns.empty:
        missing_result_df = pd.DataFrame()
        missing_column_names = missingSourceColumns['Target Field Name'].unique()
        for target_col in missing_column_names:
            transformation_values = missingSourceColumns[missingSourceColumns['Target Field Name'] == target_col][
                'Transformation Logic'].unique()
            if len(transformation_values) > 0:
                result_df[target_col] = [transformation_values[0]] * len(source_df)
            else:
                st.warning(f"No transformation logic found for {target_col}")

            

        st.write("DataFrame with Missing Columns (Filled with 'Transformation Logic' values):")
        st.dataframe(result_df)
    else:
        st.error("No missing columns found in the filtered data.")

    # Fill NaN values in the target DataFrame with 'owning_department__v' values
    if 'owning_department__v.name__v' in source_df.columns:
        if 'impacted_departments__v.name__v' in result_df.columns:
            result_df['impacted_departments__v.name__v'] = result_df['impacted_departments__v.name__v'].fillna(
                source_df['owning_department__v.name__v'])
            st.write(
                "NaN values in 'impacted_departments__v.name__v' have been filled with values from 'owning_department__v.name__v'.")
        else:
            st.warning("'impacted_departments__v.name__v' column is missing in the target data. Cannot fill NaN values.")
    else:
        st.warning("'owning_department__v.name__v' column is missing in the source data. Cannot fill NaN values.")
    if "Transformation Logic" in df.columns:
        logic_filtered = df[df["Transformation Logic"] == "Effective Date +1095"]
        date_Value = df[df["Transformation Logic"] == "Default to record creation date"]
        approved = df[df["Transformation Logic"] == "Approved Date +1095 days"]

        # Handling "Effective Date +1095" logic
        if not logic_filtered.empty:
            # Convert 'effective_date__v' to datetime, handling errors
            source_df['effective_date__v'] = pd.to_datetime(source_df['effective_date__v'], errors='coerce')

            # Copy source_df to target_df and calculate the new column
            target_df = source_df.copy()
            target_df['next_periodic_review_date__c'] = (
                    target_df['effective_date__v'] + pd.Timedelta(days=1095)
            ).dt.strftime('%Y-%m-%d')

            # Initialize result_df if it does not exist
            if 'result_df' not in globals() or not isinstance(result_df, pd.DataFrame):
                result_df = pd.DataFrame()

            # Ensure result_df has required columns
            for column in ['effective_date__v', 'next_periodic_review_date__c']:
                if column not in result_df.columns:
                    result_df[column] = None

            # Add values to result_df
            result_df['next_periodic_review_date__c'] = target_df['next_periodic_review_date__c']

        else:
            st.warning("No rows found with the specified 'Transformation Logic' value.")

        # Handling "Default to record creation date" logic
        if not date_Value.empty:
            # Add the current date as the default to 'new_draft_date__c'
            source_df['new_draft_date__c'] = datetime.now().strftime('%Y-%m-%d')

            # Update result_df with the new draft date
            if 'result_df' not in globals() or not isinstance(result_df, pd.DataFrame):
                result_df = pd.DataFrame()

            # Ensure 'new_draft_date__c' column exists
            if 'new_draft_date__c' not in result_df.columns:
                result_df['new_draft_date__c'] = None

            # Add values to result_df
            result_df['new_draft_date__c'] = source_df['new_draft_date__c']
        else:
            st.warning("No rows found with the specified 'Transformation Logic' value for default date.")
        # Handling "Approved Date +1095 days" logic
        if not approved.empty:
            # Ensure 'approved_date__vs' is in datetime format
            source_df['approved_date__vs'] = pd.to_datetime(source_df['approved_date__vs'], errors='coerce')

            # Drop rows with invalid dates, if any
            source_df = source_df.dropna(subset=['approved_date__vs'])

            # Add 1095 days to 'approved_date__vs' and create 'next_periodic_review_date__c'
            target_df['next_periodic_review_date__c'] = (
                    source_df['approved_date__vs'] + pd.Timedelta(days=1095)
            ).dt.strftime('%Y-%m-%d')

            # Initialize result_df if it does not exist
            if 'result_df' not in globals() or not isinstance(result_df, pd.DataFrame):
                result_df = pd.DataFrame()

            # Ensure result_df has required columns
            for column in ['next_periodic_review_date__c']:
                if column not in result_df.columns:
                    result_df[column] = None

            # Add values to result_df
            result_df['next_periodic_review_date__c'] = target_df['next_periodic_review_date__c']

    if 'previous_document_number__vs' in source_df.columns and 'document_number__v' in source_df.columns:
        # Both columns are available, handle NaN and create 'legacy_number__c'
        result_df['legacy_number__c'] = (
                source_df['previous_document_number__vs'].fillna('') +
                'Seres-' +
                source_df['document_number__v'].fillna('')
        )
    elif 'previous_document_number__vs' in source_df.columns:
        # Only 'previous_document_number__vs' is available, handle NaN
        result_df['legacy_number__c'] = source_df['previous_document_number__vs'].fillna('') + 'Seres-'
    elif 'document_number__v' in source_df.columns:
        # Only 'document_number__v' is available, handle NaN
        result_df['legacy_number__c'] = 'Seres-' + source_df['document_number__v'].fillna('')
    else:
        # Neither column is available
        st.warning(
            "Both 'previous_document_number__vs' and 'document_number__v' columns are missing in the source data. Cannot create 'legacy_number__c'."
        )
    if "Transformation Logic" in df.columns and "Target Field Name" in df.columns:

        # Check for 'Transformation Logic' with 'Default to record creation date'
        if "Default to record creation date" in df["Transformation Logic"].values:
            # Fill 'new_draft_date__c' with the current date where applicable
            df.loc[df[
                       "Transformation Logic"] == "Default to record creation date", "new_draft_date__c"] = datetime.now().strftime(
                '%Y-%m-%d')

            st.write(
                "Updated 'new_draft_date__c' with current date where 'Transformation Logic' is 'Default to record creation date'.")

    else:
        st.error("'Transformation Logic' or 'Target Field Name' column is missing in the dataset.")
    if "Transformation Logic" in filtered_df.columns and "Target Field Name" in filtered_df.columns:
        # Filter the target fields based on "Transformation Logic"
        target_fields = filtered_df[filtered_df["Transformation Logic"] == "Field dependency with Training Impact"][
            "Target Field Name"].tolist()

    if target_fields:
        # Check if the 'training_impact__c' column exists in source_df
        if "training_impact__vs" in source_df.columns:
            for target_field in target_fields:
                # Filter for rows where the "Target Field Name" matches
                migrate_condition = filtered_df[(filtered_df["Target Field Name"] == target_field)]
                if not migrate_condition.empty:
                    # If migration condition matches, set the corresponding target field to None
                    result_df[target_field] = None

            st.write(
                "Updated Target Fields with 'None' values where 'Transformation Logic' is 'Field dependency with Training Impact' and 'training_impact__vs' exists:")
            st.dataframe(result_df[target_fields])

    if "Transformation Logic" in df.columns and "Target Field Name" in df.columns:
        # Match the transformation logic
        target_fields = df[df["Transformation Logic"] == "Need to concatenate Seres Product: ___ Lot Number: ____"][
            "Target Field Name"].tolist()

        if target_fields:
            st.write('Target Field Names with "Need to concatenate Seres Product: ___ Lot Number: ____":')
            st.write(target_fields)

            # Check if the necessary fields exist in the "Source Field Name" column of the df
            product_exists = "product__v" in df["Source Field Name"].values
            lot_number_exists = "lot_number__c" in df["Source Field Name"].values

            if product_exists or lot_number_exists:
                # Concatenate the values according to the transformation logic
                for target_field in target_fields:
                    if product_exists and lot_number_exists:
                        result_df[target_field] = 'Seres Product:' + source_df["product__v"].astype(str) +" Lot Number:"+ source_df["lot_number__c"].astype(str)
                    elif product_exists:  # Only "product__v" exists
                        result_df[target_field] = 'Seres Product:' + source_df["product__v"].astype(str)
                    elif lot_number_exists:  # Only "lot_number__c" exists
                        result_df[target_field] = "Lot Number:" + source_df["lot_number__c"].astype(str)
                st.write("Updated Target Fields with concatenated values:")
                st.dataframe(result_df[target_fields])
            else:
                st.error(
                    "Neither 'productv' nor 'lot_numberc' field is available in the 'Source Field Name' column of the dataset.")
        else:
            st.warning(
                'No rows found with "Transformation Logic" as "Need to concatenate Seres Product: ___ Lot Number: ____".')
    else:
        st.error("'Transformation Logic' or 'Target Field Name' column is missing in the dataset.")

    st.write("Final Output DataFrame:")
    st.dataframe(result_df)

    final_output_df = result_df

    source_df.fillna('nan', inplace=True)

    picklist_file = st.file_uploader("Choose a Picklist file", type=["csv", "xlsx", "xls"],
                                     key="picklist_file")

    if picklist_file is not None:
        if picklist_file.name.endswith('.csv'):
            picklist_df = pd.read_csv(picklist_file)
        else:
            picklist_excel_file = pd.ExcelFile(picklist_file)
            picklist_sheet_name = st.selectbox("Select a sheet from Picklist",
                                               picklist_excel_file.sheet_names, key="picklist_sheet")
            picklist_df = picklist_excel_file.parse(picklist_sheet_name)

        st.write("Picklist File Preview:")
        st.dataframe(picklist_df.style.hide(axis="index"))

        picklist_df.fillna('nan', inplace=True)


        # New Code
        def map_row(row):
            match = picklist_df[
                (picklist_df['Source Document Type '] == row['type__v']) &
                (picklist_df['Source Sub Type'] == row['subtype__v']) &
                (picklist_df['Source Classification'] == row['classification__v']) &
                (picklist_df['Lifecycle'] == row['lifecycle__v'])
                ]
            if not match.empty:
                return pd.Series({
                    'type__v': match['Target Document Type '].values[0],
                    'subtype__v': match['Target Sub Type'].values[0],
                    'classification__v': match['Target Classification'].values[0],
                    'lifecycle__v': match['Target Lifecycle'].values[0]
                })
            return pd.Series(
                {'type__v': np.nan, 'subtype__v': np.nan, 'classification__v': np.nan, 'lifecycle__v': np.nan})


        mapped_df = source_df.apply(map_row, axis=1)
        st.write("Mapped DataFrame:")
        st.dataframe(mapped_df)

        # Update final_output_df with mapped_df values if columns match
        for column in mapped_df.columns:
            if column in final_output_df.columns:
                final_output_df[column] = mapped_df[column]

        # Display the updated final_output_df
        st.write("Updated Output DataFrame:")
        st.dataframe(final_output_df)

        conditions2 = [
            source_df['status__v'] == 'Retired',
            source_df['status__v'] == 'Obsolete',
            source_df['status__v'] == 'Superseded',
            source_df['status__v'] == 'Final',
            source_df['status__v'] == 'Archived',
            source_df['status__v'] == 'Approved',
            source_df['status__v'] == 'Effective'
        ]
        choices2 = ['Obsolete', 'Obsolete', 'Superseded', 'Final', 'Final', 'Final', 'Final']
        final_output_df['status__v'] = np.select(conditions2, choices2, default='')

        st.dataframe(final_output_df)

        # Ensure all columns in target_field_order exist in final_output_df
        target_field_order = df['Target Field Name'].unique()
        for col in target_field_order:
            if col not in final_output_df.columns:
                final_output_df[col] = ""  # Add missing columns with default empty values

        # Reorder the columns based on target_field_order
        final_output_df = final_output_df[target_field_order]

        # Update the classification__v column, where np.nan becomes blank
        final_output_df['classification__v'] = final_output_df['classification__v'].replace('nan', '')
        final_output_df['subtype__v'] = final_output_df['subtype__v'].replace('nan', '')

        final_output_df = final_output_df.loc[:, ~final_output_df.columns.isna()]

        st.write("Final Output DataFrame (Ordered by Target Field Name):")
        st.dataframe(final_output_df.style.hide(axis="index"))

        # Prepare final CSV for download
        final_output_csv = final_output_df.to_csv(index=False)
        st.download_button(
            label="Download Final Output CSV File",
            data=final_output_csv,
            file_name="final_output_ordered.csv",
            mime="text/csv"
        )