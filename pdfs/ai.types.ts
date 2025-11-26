export type AiInvoice = {
  invoice_currency?: string | null;
  invoice_customer_address?: string | null;
  invoice_customer_country?: string | null;
  invoice_customer_name?: string | null;
  invoice_date?: string | null;
  invoice_delivery_term?: string | null;
  invoice_id?: string | null;
  invoice_payment_term?: string | null;
  invoice_po_number?: string | null;
  invoice_shipment_country_of_origin?: string | null;
  invoice_supplier_address?: string | null;
  invoice_supplier_country?: string | null;
  invoice_supplier_name?: string | null;
  invoice_total_amount?: number | null;
  invoice_total_package_quantity?: number | null;
  invoice_total_quantity?: number | null;
  items?:
    | {
        invoice_item_commodity_code?: string | null;
        invoice_item_country_of_origin?: string | null;
        invoice_item_description?: string | null;
        invoice_item_no?: number | null;
        invoice_item_package_quantity?: number | null;
        invoice_item_product_id?: string | null;
        invoice_item_quantity?: number | null;
        invoice_item_total_amount?: number | null;
        invoice_item_unit_price?: number | null;
        invoice_item_unit_type?: string | null;
      }[]
    | null;
}[];