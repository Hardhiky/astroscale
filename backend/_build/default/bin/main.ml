open Lwt.Infix
open Cohttp
open Cohttp_lwt_unix

let port = 8080

let () = Printf.printf "Starting server on http://localhost:%d\n%!" port

let cors_headers =
  Header.of_list
    [ ("Access-Control-Allow-Origin", "*")
    ; ("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    ; ("Access-Control-Allow-Headers", "Content-Type")
    ]
;;

(* Helper function to convert JSON number to float, handling both int and float *)
let to_float_flexible json =
  try Yojson.Basic.Util.to_float json
  with Yojson.Basic.Util.Type_error _ ->
    float_of_int (Yojson.Basic.Util.to_int json)
;;

let predict_handler _body =
  Lwt.catch
    (fun () ->
      let open Yojson.Basic.Util in
      Cohttp_lwt.Body.to_string _body >>= fun body ->
      Printf.printf "Received body: %s\n%!" body;
      let json = Yojson.Basic.from_string body in
      let ra = json |> member "ra" |> to_float_flexible in
      let dec = json |> member "dec" |> to_float_flexible in
      let teff = json |> member "teff" |> to_float_flexible in
      let logg = json |> member "logg" |> to_float_flexible in
      let fe_h = json |> member "fe_h" |> to_float_flexible in
      let snr = json |> member "snr" |> to_float_flexible in
      let parallax = json |> member "parallax" |> to_float_flexible in

      let cmd =
        Printf.sprintf
          "cd .. && ~/.virtualenvs/py3.11/bin/python inference_production.py %.6f %.6f %.6f %.6f %.6f %.6f %.6f 2>&1"
          ra dec teff logg fe_h snr parallax
      in
      Printf.printf "Executing: %s\n%!" cmd;
      let ic = Unix.open_process_in cmd in
      (* Read all output lines to capture both stdout and stderr *)
      let rec read_lines acc =
        try
          let line = input_line ic in
          read_lines (line :: acc)
        with End_of_file -> List.rev acc
      in
      let all_output = read_lines [] in
      let status = Unix.close_process_in ic in

      (* Print all output for debugging *)
      List.iter (fun line -> Printf.printf "Python: %s\n%!" line) all_output;
      Printf.printf "Python exit status: %s\n%!"
        (match status with
        | Unix.WEXITED n -> Printf.sprintf "exit %d" n
        | Unix.WSIGNALED n -> Printf.sprintf "signal %d" n
        | Unix.WSTOPPED n -> Printf.sprintf "stopped %d" n);

      (* Get the last line as the actual output (should be the z value) *)
      let output =
        match List.rev all_output with
        | last :: _ -> last
        | [] -> raise (Failure "No output from Python script")
      in

      (* Parse the float value from Python output *)
      let z_value = float_of_string (String.trim output) in
      let resp =
        `Assoc [ ("predicted_z", `Float z_value) ]
        |> Yojson.Basic.to_string
      in
      Server.respond_string ~status:`OK ~headers:cors_headers ~body:resp ())
    (fun exn ->
      let error_msg = Printexc.to_string exn in
      Printf.printf "Error in predict_handler: %s\n%!" error_msg;
      let error_resp =
        `Assoc [ ("error", `String error_msg) ]
        |> Yojson.Basic.to_string
      in
      Server.respond_string ~status:`Internal_server_error ~headers:cors_headers ~body:error_resp ())
;;

let server =
  let callback _conn req body =
    let path = Uri.path (Request.uri req) in
    let meth = Request.meth req in
    Printf.printf "[%s] %s\n%!" (Code.string_of_method meth) path;
    match (meth, path) with
    | `OPTIONS, _ -> Server.respond_string ~status:`OK ~headers:cors_headers ~body:"" ()
    | `POST, "/predict" -> predict_handler body
    | _ -> Server.respond_string ~status:`Not_found ~headers:cors_headers ~body:"Not Found" ()
  in
  Server.create ~mode:(`TCP (`Port port)) (Server.make ~callback ())
;;

let () = Lwt_main.run server
